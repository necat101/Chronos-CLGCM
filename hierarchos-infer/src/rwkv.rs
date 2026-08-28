use crate::format::TensorMap;
use crate::math::{
    clamp_finite, group_norm_heads, layer_norm_affine, sigmoid, softplus, Linear, Matrix,
    LAYER_NORM_EPS,
};
use crate::{Error, Result, Tensor};

const GROUP_NORM_EPS: f32 = 64.0e-5;

#[derive(Debug, Clone)]
pub(crate) struct RwkvState {
    pub prev_tm: Vec<f32>,
    pub prev_cm: Vec<f32>,
    pub v_first: Vec<f32>,
    pub output: Vec<f32>,
    /// Packed [head, row, column] matrix state.
    pub matrix: Vec<f32>,
}

impl RwkvState {
    pub fn zeros(width: usize, head_size: usize) -> Self {
        Self {
            prev_tm: vec![0.0; width],
            prev_cm: vec![0.0; width],
            v_first: vec![0.0; width],
            output: vec![0.0; width],
            matrix: vec![0.0; width * head_size],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateReadout {
    LegacyInputCache,
    ExplicitOutput,
}

#[derive(Debug, Clone)]
pub(crate) struct RwkvCell {
    width: usize,
    heads: usize,
    head_size: usize,
    readout: StateReadout,
    state_clamp: f32,
    channel_mix_key_clamp: f32,
    channel_mix_deepembed_clamp: f32,

    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
    ln_x_weight: Vec<f32>,
    ln_x_bias: Vec<f32>,

    x_r: Vec<f32>,
    x_w: Vec<f32>,
    x_k: Vec<f32>,
    x_v: Vec<f32>,
    x_a: Vec<f32>,
    x_g: Vec<f32>,
    x_k_cm: Vec<f32>,

    w0: Vec<f32>,
    w1: Matrix,
    w2: Matrix,
    a0: Vec<f32>,
    a1: Matrix,
    a2: Matrix,
    g1: Matrix,
    g2: Matrix,
    k_k: Vec<f32>,
    k_a: Vec<f32>,
    r_k: Vec<f32>,

    receptance: Linear,
    key: Linear,
    value: Linear,
    output_projection: Linear,
    key_cm: Linear,
    value_cm: Linear,
}

impl RwkvCell {
    pub fn load(
        map: &mut TensorMap,
        prefix: &str,
        width: usize,
        requested_head_size: Option<usize>,
        state_readout_mode: &str,
        state_clamp: f32,
        channel_mix_key_clamp: f32,
        channel_mix_deepembed_clamp: f32,
    ) -> Result<Self> {
        let r_k_name = format!("{prefix}.r_k");
        let r_k_tensor = map.take(&r_k_name)?;
        if r_k_tensor.dims.len() != 2 {
            return Err(Error::Shape {
                name: r_k_name,
                expected: vec![0, 0],
                actual: r_k_tensor.dims,
            });
        }
        let heads = r_k_tensor.dims[0];
        let head_size = r_k_tensor.dims[1];
        if heads == 0 || head_size == 0 || heads * head_size != width {
            return Err(Error::InvalidConfig(format!(
                "{prefix}.r_k geometry [{heads}, {head_size}] does not cover hidden width {width}"
            )));
        }
        if let Some(requested) = requested_head_size {
            if requested != head_size {
                return Err(Error::InvalidConfig(format!(
                    "{prefix} config head size {requested} disagrees with checkpoint head size {head_size}"
                )));
            }
        }
        let readout = match state_readout_mode {
            "legacy-input-cache" => StateReadout::LegacyInputCache,
            "explicit-output" => StateReadout::ExplicitOutput,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported RWKV state readout mode {other:?}"
                )))
            }
        };

        let ln1_weight = take_vector(map, &format!("{prefix}.ln1.weight"), width)?;
        let ln1_bias = take_vector(map, &format!("{prefix}.ln1.bias"), width)?;
        let ln2_weight = take_vector(map, &format!("{prefix}.ln2.weight"), width)?;
        let ln2_bias = take_vector(map, &format!("{prefix}.ln2.bias"), width)?;
        let ln_x_weight = take_vector(map, &format!("{prefix}.ln_x.weight"), width)?;
        let ln_x_bias = take_vector(map, &format!("{prefix}.ln_x.bias"), width)?;

        let x_r = take_vector_relaxed(map, &format!("{prefix}.x_r"), width)?;
        let x_w = take_vector_relaxed(map, &format!("{prefix}.x_w"), width)?;
        let x_k = take_vector_relaxed(map, &format!("{prefix}.x_k"), width)?;
        let x_v = take_vector_relaxed(map, &format!("{prefix}.x_v"), width)?;
        let x_a = take_vector_relaxed(map, &format!("{prefix}.x_a"), width)?;
        let x_g = take_vector_relaxed(map, &format!("{prefix}.x_g"), width)?;
        let x_k_cm = take_vector_relaxed(map, &format!("{prefix}.x_k_cm"), width)?;
        let w0 = take_vector_relaxed(map, &format!("{prefix}.w0"), width)?;
        let a0 = take_vector_relaxed(map, &format!("{prefix}.a0"), width)?;
        let k_k = take_vector_relaxed(map, &format!("{prefix}.k_k"), width)?;
        let k_a = take_vector_relaxed(map, &format!("{prefix}.k_a"), width)?;

        let w1 = take_matrix_unknown_cols(map, &format!("{prefix}.w1"), width)?;
        let w2 = take_matrix(map, &format!("{prefix}.w2"), w1.cols, width)?;
        let a1 = take_matrix_unknown_cols(map, &format!("{prefix}.a1"), width)?;
        let a2 = take_matrix(map, &format!("{prefix}.a2"), a1.cols, width)?;
        let g1 = take_matrix_unknown_cols(map, &format!("{prefix}.g1"), width)?;
        let g2 = take_matrix(map, &format!("{prefix}.g2"), g1.cols, width)?;

        let receptance = take_linear(map, &format!("{prefix}.receptance"), width, width, false)?;
        let key = take_linear(map, &format!("{prefix}.key"), width, width, false)?;
        let value = take_linear(map, &format!("{prefix}.value"), width, width, false)?;
        let output_projection = take_linear(map, &format!("{prefix}.output"), width, width, false)?;
        let key_cm = take_linear(map, &format!("{prefix}.key_cm"), width * 4, width, false)?;
        let value_cm = take_linear(map, &format!("{prefix}.value_cm"), width, width * 4, false)?;

        Ok(Self {
            width,
            heads,
            head_size,
            readout,
            state_clamp,
            channel_mix_key_clamp,
            channel_mix_deepembed_clamp,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            ln_x_weight,
            ln_x_bias,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            x_k_cm,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            g1,
            g2,
            k_k,
            k_a,
            r_k: r_k_tensor.data,
            receptance,
            key,
            value,
            output_projection,
            key_cm,
            value_cm,
        })
    }

    pub fn initial_state(&self) -> RwkvState {
        RwkvState::zeros(self.width, self.head_size)
    }

    pub fn state_hidden<'a>(&self, state: &'a RwkvState) -> &'a [f32] {
        match self.readout {
            StateReadout::LegacyInputCache => &state.prev_tm,
            StateReadout::ExplicitOutput => &state.output,
        }
    }

    pub fn forward(
        &self,
        x: &[f32],
        state: &RwkvState,
        deepembed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, RwkvState)> {
        if x.len() != self.width {
            return Err(Error::InvalidConfig(format!(
                "RWKV input width {} does not match {}",
                x.len(),
                self.width
            )));
        }
        if state.prev_tm.len() != self.width
            || state.prev_cm.len() != self.width
            || state.matrix.len() != self.width * self.head_size
        {
            return Err(Error::InvalidConfig("RWKV state geometry mismatch".into()));
        }
        if let Some(deepembed) = deepembed {
            if deepembed.len() != self.width * 4 {
                return Err(Error::InvalidConfig(format!(
                    "RWKV DeepEmbed width {} does not match {}",
                    deepembed.len(),
                    self.width * 4
                )));
            }
        }

        let mut x_norm = Vec::with_capacity(self.width);
        layer_norm_affine(
            x,
            &self.ln1_weight,
            &self.ln1_bias,
            LAYER_NORM_EPS,
            &mut x_norm,
        );

        let xr = mix(&x_norm, &state.prev_tm, &self.x_r);
        let xw = mix(&x_norm, &state.prev_tm, &self.x_w);
        let xk = mix(&x_norm, &state.prev_tm, &self.x_k);
        let xv = mix(&x_norm, &state.prev_tm, &self.x_v);
        let xa = mix(&x_norm, &state.prev_tm, &self.x_a);
        let xg = mix(&x_norm, &state.prev_tm, &self.x_g);

        let mut r = Vec::new();
        let mut k = Vec::new();
        let mut v = Vec::new();
        self.receptance.forward(&xr, &mut r);
        self.key.forward(&xk, &mut k);
        self.value.forward(&xv, &mut v);
        let v_first = v.clone();

        let mut tmp_rank = Vec::new();
        let mut tmp_width = Vec::new();
        self.a1.row_vector_mul(&xa, &mut tmp_rank);
        self.a2.row_vector_mul(&tmp_rank, &mut tmp_width);
        let mut a = Vec::with_capacity(self.width);
        for i in 0..self.width {
            a.push(sigmoid(self.a0[i] + tmp_width[i]));
        }

        self.g1.row_vector_mul(&xg, &mut tmp_rank);
        for value in &mut tmp_rank {
            *value = sigmoid(*value);
        }
        self.g2.row_vector_mul(&tmp_rank, &mut tmp_width);
        let g = tmp_width.clone();

        self.w1.row_vector_mul(&xw, &mut tmp_rank);
        for value in &mut tmp_rank {
            *value = value.tanh();
        }
        self.w2.row_vector_mul(&tmp_rank, &mut tmp_width);
        let mut w_decay = vec![0.0f32; self.width];
        for i in 0..self.width {
            let inner = self.w0[i] + tmp_width[i];
            let w = -softplus(-inner) - 0.5;
            let bounded = w.clamp(-60.0, 30.0);
            w_decay[i] = (-bounded.exp()).exp();
        }

        let mut kk = vec![0.0f32; self.width];
        for head in 0..self.heads {
            let start = head * self.head_size;
            let mut norm_sq = 0.0f32;
            for j in 0..self.head_size {
                let index = start + j;
                kk[index] = k[index] * self.k_k[index];
                norm_sq += kk[index] * kk[index];
            }
            // torch.nn.functional.normalize defaults eps=1e-12 and divides by
            // max(norm, eps).
            let denom = norm_sq.sqrt().max(1.0e-12);
            for j in 0..self.head_size {
                kk[start + j] /= denom;
            }
        }
        for i in 0..self.width {
            k[i] *= 1.0 + (a[i] - 1.0) * self.k_a[i];
        }

        let mut matrix = state.matrix.clone();
        let mut tmix = vec![0.0f32; self.width];
        for head in 0..self.heads {
            let channel = head * self.head_size;
            let matrix_base = head * self.head_size * self.head_size;
            let mut sa = vec![0.0f32; self.head_size];
            for row in 0..self.head_size {
                let base = matrix_base + row * self.head_size;
                let a_head = &kk[channel..channel + self.head_size];
                // state_a = -kk
                sa[row] = -crate::math::dot(&matrix[base..base + self.head_size], a_head);
            }
            for row in 0..self.head_size {
                let row_channel = channel + row;
                let base = matrix_base + row * self.head_size;
                for col in 0..self.head_size {
                    let col_channel = channel + col;
                    let b = kk[col_channel] * a[col_channel];
                    matrix[base + col] = matrix[base + col] * w_decay[col_channel]
                        + sa[row] * b
                        + v[row_channel] * k[col_channel];
                }
                tmix[row_channel] = crate::math::dot(
                    &matrix[base..base + self.head_size],
                    &r[channel..channel + self.head_size],
                );
            }
        }

        let mut tmix_norm = Vec::new();
        group_norm_heads(
            &tmix,
            self.heads,
            &self.ln_x_weight,
            &self.ln_x_bias,
            GROUP_NORM_EPS,
            &mut tmix_norm,
        );
        for head in 0..self.heads {
            let start = head * self.head_size;
            let mut bonus_scale = 0.0f32;
            for j in 0..self.head_size {
                let index = start + j;
                bonus_scale += r[index] * k[index] * self.r_k[index];
            }
            for j in 0..self.head_size {
                let index = start + j;
                tmix_norm[index] += bonus_scale * v[index];
                tmix_norm[index] *= g[index];
            }
        }

        let mut projected = Vec::new();
        self.output_projection.forward(&tmix_norm, &mut projected);
        let mut x_after_tm = vec![0.0f32; self.width];
        for i in 0..self.width {
            x_after_tm[i] = x[i] + projected[i];
        }

        let mut x_norm2 = Vec::new();
        layer_norm_affine(
            &x_after_tm,
            &self.ln2_weight,
            &self.ln2_bias,
            LAYER_NORM_EPS,
            &mut x_norm2,
        );
        let xk_cm = mix(&x_norm2, &state.prev_cm, &self.x_k_cm);
        let mut cm_key = Vec::new();
        self.key_cm.forward(&xk_cm, &mut cm_key);
        if self.channel_mix_key_clamp > 0.0 {
            clamp_finite(&mut cm_key, self.channel_mix_key_clamp)?;
        }
        for value in &mut cm_key {
            *value = value.max(0.0);
            *value *= *value;
        }
        if let Some(deep) = deepembed {
            let ffn_limit = self.channel_mix_key_clamp
                * self.channel_mix_key_clamp
                * self.channel_mix_deepembed_clamp;
            for (value, &modulation) in cm_key.iter_mut().zip(deep) {
                if !modulation.is_finite() {
                    return Err(Error::NonFinite("RWKV DeepEmbed".into()));
                }
                let modulation = if self.channel_mix_deepembed_clamp > 0.0 {
                    modulation.clamp(
                        -self.channel_mix_deepembed_clamp,
                        self.channel_mix_deepembed_clamp,
                    )
                } else {
                    modulation
                };
                *value *= modulation;
                if ffn_limit > 0.0 {
                    *value = value.clamp(-ffn_limit, ffn_limit);
                }
            }
        }
        let mut cm_projected = Vec::new();
        self.value_cm.forward(&cm_key, &mut cm_projected);
        let mut output = vec![0.0f32; self.width];
        for i in 0..self.width {
            output[i] = x_after_tm[i] + cm_projected[i];
        }

        let mut new_state = RwkvState {
            prev_tm: x_norm,
            prev_cm: x_norm2,
            v_first,
            output: output.clone(),
            matrix,
        };
        if self.state_clamp > 0.0 {
            clamp_finite(&mut new_state.prev_tm, self.state_clamp)?;
            clamp_finite(&mut new_state.prev_cm, self.state_clamp)?;
            clamp_finite(&mut new_state.v_first, self.state_clamp)?;
            clamp_finite(&mut new_state.output, self.state_clamp)?;
            clamp_finite(&mut new_state.matrix, self.state_clamp)?;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("RWKV output".into()));
        }
        Ok((output, new_state))
    }
}

fn mix(x: &[f32], previous: &[f32], coeff: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(previous)
        .zip(coeff)
        .map(|((&current, &old), &c)| current + (old - current) * c)
        .collect()
}

fn take_vector(map: &mut TensorMap, name: &str, width: usize) -> Result<Vec<f32>> {
    Ok(map.take_shape(name, &[width])?.data)
}

fn take_vector_relaxed(map: &mut TensorMap, name: &str, width: usize) -> Result<Vec<f32>> {
    let tensor = map.take(name)?;
    let shape_ok = tensor.dims.as_slice() == [width] || tensor.dims.as_slice() == [1, width];
    if tensor.data.len() != width || !shape_ok {
        return Err(Error::Shape {
            name: name.into(),
            expected: vec![1, width],
            actual: tensor.dims,
        });
    }
    Ok(tensor.data)
}

fn take_matrix(map: &mut TensorMap, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
    Matrix::from_tensor(name, map.take(name)?, rows, cols)
}

fn take_matrix_unknown_cols(map: &mut TensorMap, name: &str, rows: usize) -> Result<Matrix> {
    let tensor = map.take(name)?;
    if tensor.dims.len() != 2 || tensor.dims[0] != rows || tensor.dims[1] == 0 {
        return Err(Error::Shape {
            name: name.into(),
            expected: vec![rows, 0],
            actual: tensor.dims,
        });
    }
    let cols = tensor.dims[1];
    Matrix::from_tensor(name, tensor, rows, cols)
}

pub(crate) fn take_linear(
    map: &mut TensorMap,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    bias: bool,
) -> Result<Linear> {
    let weight_name = format!("{prefix}.weight");
    let weight = Matrix::from_tensor(
        &weight_name,
        map.take(&weight_name)?,
        out_features,
        in_features,
    )?;
    let bias_tensor = if bias {
        Some(map.take(&format!("{prefix}.bias"))?)
    } else {
        None
    };
    Linear::new(weight, bias_tensor, prefix)
}

pub(crate) fn take_layer_norm(
    map: &mut TensorMap,
    prefix: &str,
    width: usize,
) -> Result<crate::math::LayerNorm> {
    Ok(crate::math::LayerNorm {
        weight: take_vector(map, &format!("{prefix}.weight"), width)?,
        bias: take_vector(map, &format!("{prefix}.bias"), width)?,
        eps: LAYER_NORM_EPS,
    })
}

#[allow(dead_code)]
fn _tensor_data(tensor: Tensor) -> Vec<f32> {
    tensor.data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_geometry_matches_packed_python_layout() {
        let state = RwkvState::zeros(12, 4);
        assert_eq!(state.matrix.len(), 48);
    }
}
