use crate::{Error, Result, Tensor};

pub(crate) const LAYER_NORM_EPS: f32 = 1.0e-5;

#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sums = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let p = i * 4;
        sums[0] += a[p] * b[p];
        sums[1] += a[p + 1] * b[p + 1];
        sums[2] += a[p + 2] * b[p + 2];
        sums[3] += a[p + 3] * b[p + 3];
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn from_tensor(name: &str, tensor: Tensor, rows: usize, cols: usize) -> Result<Self> {
        if tensor.dims != [rows, cols] {
            return Err(Error::Shape {
                name: name.into(),
                expected: vec![rows, cols],
                actual: tensor.dims,
            });
        }
        Ok(Self {
            rows,
            cols,
            data: tensor.data,
        })
    }

    #[inline]
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// PyTorch-style linear layer multiplication: y = x W^T.
    pub fn linear(&self, x: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), self.cols);
        out.clear();
        out.reserve(self.rows);
        for row in 0..self.rows {
            out.push(dot(x, self.row(row)));
        }
    }

    /// Parameter-matrix multiplication used by RWKV LoRA factors: y = x A.
    /// Here the stored tensor shape is [input, output].
    pub fn row_vector_mul(&self, x: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), self.rows);
        out.clear();
        out.resize(self.cols, 0.0);
        for (r, &xr) in x.iter().enumerate() {
            let row = self.row(r);
            for c in 0..self.cols {
                out[c] += xr * row[c];
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Linear {
    pub weight: Matrix,
    pub bias: Option<Vec<f32>>,
}

impl Linear {
    pub fn new(weight: Matrix, bias: Option<Tensor>, name: &str) -> Result<Self> {
        let bias = match bias {
            Some(t) => {
                if t.dims != [weight.rows] {
                    return Err(Error::Shape {
                        name: format!("{name}.bias"),
                        expected: vec![weight.rows],
                        actual: t.dims,
                    });
                }
                Some(t.data)
            }
            None => None,
        };
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, x: &[f32], out: &mut Vec<f32>) {
        self.weight.linear(x, out);
        if let Some(bias) = &self.bias {
            for (value, &b) in out.iter_mut().zip(bias) {
                *value += b;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &[f32], out: &mut Vec<f32>) {
        layer_norm_affine(x, &self.weight, &self.bias, self.eps, out)
    }
}

#[inline]
pub(crate) fn layer_norm_affine(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), bias.len());
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (variance + eps).sqrt();
    out.clear();
    out.reserve(x.len());
    for i in 0..x.len() {
        out.push((x[i] - mean) * inv_std * weight[i] + bias[i]);
    }
}

#[inline]
pub(crate) fn layer_norm_no_affine(x: &[f32], eps: f32, out: &mut Vec<f32>) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    let inv_std = 1.0 / (variance + eps).sqrt();
    out.clear();
    out.reserve(x.len());
    for &v in x {
        out.push((v - mean) * inv_std);
    }
}

pub(crate) fn group_norm_heads(
    x: &[f32],
    heads: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), bias.len());
    debug_assert_eq!(x.len() % heads, 0);
    let width = x.len() / heads;
    out.clear();
    out.resize(x.len(), 0.0);
    for head in 0..heads {
        let start = head * width;
        let values = &x[start..start + width];
        let mean = values.iter().sum::<f32>() / width as f32;
        let variance = values
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / width as f32;
        let inv_std = 1.0 / (variance + eps).sqrt();
        for i in 0..width {
            let index = start + i;
            out[index] = (x[index] - mean) * inv_std * weight[index] + bias[index];
        }
    }
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub(crate) fn clamp_finite(values: &mut [f32], max_abs: f32) -> Result<()> {
    if max_abs <= 0.0 {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("activation clamp".into()));
        }
        return Ok(());
    }
    for value in values {
        if !value.is_finite() {
            return Err(Error::NonFinite("activation clamp".into()));
        }
        *value = value.clamp(-max_abs, max_abs);
    }
    Ok(())
}

pub(crate) fn l2_norm_clamp(values: &mut [f32], max_norm: f32) -> Result<()> {
    clamp_finite(values, f32::MAX)?;
    if max_norm <= 0.0 {
        return Ok(());
    }
    let norm_sq = values.iter().map(|v| v * v).sum::<f32>();
    if norm_sq > max_norm * max_norm {
        let scale = max_norm / norm_sq.sqrt();
        for value in values {
            *value *= scale;
        }
    }
    Ok(())
}

pub(crate) fn softmax_in_place(values: &mut [f32]) -> Result<()> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(Error::NonFinite("softmax input".into()));
    }
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::NonFinite("softmax normalization".into()));
    }
    let inv = 1.0 / sum;
    for value in values {
        *value *= inv;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_uses_population_variance() {
        let mut out = Vec::new();
        layer_norm_no_affine(&[1.0, 3.0], 0.0, &mut out);
        assert!((out[0] + 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stable_sigmoid_extremes() {
        assert!(sigmoid(100.0) > 0.9999);
        assert!(sigmoid(-100.0) < 0.0001);
    }
}
