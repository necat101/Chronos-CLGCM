use std::collections::HashSet;
use std::path::Path;

use crate::format::{ModelFile, TensorMap};
use crate::ltm::Ltm;
use crate::math::{
    clamp_finite, gelu, l2_norm_clamp, layer_norm_no_affine, sigmoid, silu, softmax_in_place,
    LayerNorm, Linear, Matrix, LAYER_NORM_EPS,
};
use crate::rosa::RosaState;
use crate::rwkv::{take_layer_norm, take_linear, RwkvCell, RwkvState};
use crate::{Error, ModelConfig, Result};

#[derive(Debug, Clone)]
struct TokenAdapter {
    input_dim: usize,
    output_dim: usize,
    down: Linear,
    up: Linear,
    bias: Vec<f32>,
}

impl TokenAdapter {
    fn load(
        map: &mut TensorMap,
        prefix: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let down_tensor = map.take(&format!("{prefix}.down.weight"))?;
        if down_tensor.dims.len() != 2 || down_tensor.dims[1] != input_dim {
            return Err(Error::Shape {
                name: format!("{prefix}.down.weight"),
                expected: vec![0, input_dim],
                actual: down_tensor.dims,
            });
        }
        let rank = down_tensor.dims[0];
        let down = Linear::new(
            Matrix::from_tensor(
                &format!("{prefix}.down.weight"),
                down_tensor,
                rank,
                input_dim,
            )?,
            None,
            &format!("{prefix}.down"),
        )?;
        let up = take_linear(map, &format!("{prefix}.up"), output_dim, rank, false)?;
        let bias_tensor = map.take_shape(&format!("{prefix}.bias"), &[output_dim])?;
        Ok(Self {
            input_dim,
            output_dim,
            down,
            up,
            bias: bias_tensor.data,
        })
    }

    fn forward(&self, token_features: &[f32]) -> Result<Vec<f32>> {
        let mut normalized = Vec::new();
        layer_norm_no_affine(token_features, LAYER_NORM_EPS, &mut normalized);
        self.forward_normalized(&normalized)
    }

    fn forward_normalized(&self, normalized: &[f32]) -> Result<Vec<f32>> {
        if normalized.len() != self.input_dim {
            return Err(Error::InvalidConfig(format!(
                "token adapter input width {} does not match {}",
                normalized.len(),
                self.input_dim
            )));
        }
        let mut hidden = Vec::new();
        self.down.forward(normalized, &mut hidden);
        for value in &mut hidden {
            *value = silu(*value);
        }
        let mut out = Vec::new();
        self.up.forward(&hidden, &mut out);
        debug_assert_eq!(out.len(), self.output_dim);
        for (value, &bias) in out.iter_mut().zip(&self.bias) {
            *value += bias;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
enum DeepEmbed {
    Off,
    Legacy {
        h_table: Matrix,
        l_table: Matrix,
    },
    Shared {
        h_adapter: TokenAdapter,
        l_adapter: TokenAdapter,
    },
}

#[derive(Debug, Clone)]
enum RosaEmbedding {
    Off,
    Legacy(Matrix),
    Shared(TokenAdapter),
}

#[derive(Debug, Clone)]
pub struct InferenceState {
    pub(crate) h_state: RwkvState,
    pub(crate) l_state: RwkvState,
    pub prev_context: Vec<f32>,
    pub target_context: Vec<f32>,
    pub drift_state: Vec<f32>,
    pub global_pos: usize,
    rosa: Option<RosaState>,
    /// Reserved for future opt-in online memory writes. `None` is the fast,
    /// read-only inference path and matches ordinary generation with suppressed
    /// Hebbian updates.
    ltm_fast_vals: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub logits: Vec<f32>,
    pub topk_indices: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 0x4849_4552_4152_4348,
        }
    }
}

#[derive(Debug)]
pub struct Hierarchos {
    config: ModelConfig,
    token_embedding: Matrix,
    persistent: Vec<f32>,
    memory_gate_warmup_step: f32,
    ltm_gate_logit: f32,
    rosa_gate_logit: Option<f32>,
    ltm_router: Option<Linear>,
    rosa_router: Option<Linear>,
    deepembed: DeepEmbed,
    rosa_embedding: RosaEmbedding,
    ltm: Ltm,
    qproj: Linear,
    in_proj: Linear,
    l_feedback_proj: Linear,
    h_rnn: RwkvCell,
    h_to_context: Linear,
    h_halt_proj: Linear,
    l_input_proj: Linear,
    l_rnn: RwkvCell,
    context_drift_proj: Linear,
    l_to_out: Linear,
    out_norm: LayerNorm,
}

impl Hierarchos {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let ModelFile { config, tensors } = ModelFile::load(path)?;
        Self::from_parts(config, TensorMap(tensors))
    }

    fn from_parts(config: ModelConfig, mut map: TensorMap) -> Result<Self> {
        config.validate()?;
        if config.h_hidden != config.context_dim {
            return Err(Error::InvalidConfig(format!(
                "h_hidden ({}) must equal context_dim ({}) for manager residuals",
                config.h_hidden, config.context_dim
            )));
        }

        let token_embedding = Matrix::from_tensor(
            "tok_emb.weight",
            map.take("tok_emb.weight")?,
            config.vocab_size,
            config.context_dim,
        )?;
        let persistent = map.take_shape("persistent", &[config.persistent_dim])?.data;
        let memory_gate_warmup_step = map
            .take_optional("memory_gate_warmup_step")
            .map(|t| t.scalar("memory_gate_warmup_step"))
            .transpose()?
            .unwrap_or(0.0);
        let ltm_gate_logit = map.take_scalar("ltm_gate_logit")?;

        let deepembed = if !config.use_deepembed || config.deepembed_mode == "off" {
            DeepEmbed::Off
        } else if config.deepembed_mode == "legacy-table" {
            DeepEmbed::Legacy {
                h_table: Matrix::from_tensor(
                    "h_deepemb.weight",
                    map.take("h_deepemb.weight")?,
                    config.vocab_size,
                    config.h_hidden * 4,
                )?,
                l_table: Matrix::from_tensor(
                    "l_deepemb.weight",
                    map.take("l_deepemb.weight")?,
                    config.vocab_size,
                    config.l_hidden * 4,
                )?,
            }
        } else {
            DeepEmbed::Shared {
                h_adapter: TokenAdapter::load(
                    &mut map,
                    "h_deepembed_adapter",
                    config.context_dim,
                    config.h_hidden * 4,
                )?,
                l_adapter: TokenAdapter::load(
                    &mut map,
                    "l_deepembed_adapter",
                    config.context_dim,
                    config.l_hidden * 4,
                )?,
            }
        };

        let (rosa_embedding, rosa_gate_logit) =
            if !config.use_rosa || config.rosa_embedding_mode == "off" {
                (RosaEmbedding::Off, None)
            } else {
                let gate = map.take_scalar("rosa_gate_logit")?;
                let embedding = if config.rosa_embedding_mode == "legacy-table" {
                    RosaEmbedding::Legacy(Matrix::from_tensor(
                        "rosa_emb.weight",
                        map.take("rosa_emb.weight")?,
                        config.vocab_size + 1,
                        config.context_dim,
                    )?)
                } else {
                    RosaEmbedding::Shared(TokenAdapter::load(
                        &mut map,
                        "rosa_adapter",
                        config.context_dim,
                        config.context_dim,
                    )?)
                };
                (embedding, Some(gate))
            };

        let ltm_router = if config.memory_token_routers {
            Some(take_linear(
                &mut map,
                "ltm_router",
                1,
                config.context_dim,
                true,
            )?)
        } else {
            None
        };
        let rosa_router = if config.memory_token_routers && config.use_rosa {
            Some(take_linear(
                &mut map,
                "rosa_router",
                1,
                config.context_dim,
                true,
            )?)
        } else {
            None
        };

        let ltm = Ltm::load(
            &mut map,
            config.ltm_slots,
            config.ltm_key_dim,
            config.ltm_val_dim,
        )?;
        let qproj = take_linear(
            &mut map,
            "qproj",
            config.ltm_key_dim,
            config.context_dim * 2,
            false,
        )?;
        let mac_width =
            config.context_dim + config.persistent_dim + config.ltm_val_dim * config.ltm_topk;
        let in_proj = take_linear(&mut map, "in_proj", config.context_dim, mac_width, true)?;
        let l_feedback_proj = take_linear(
            &mut map,
            "l_feedback_proj",
            config.h_hidden,
            config.l_hidden,
            false,
        )?;
        let h_rnn = RwkvCell::load(
            &mut map,
            "h_rnn",
            config.h_hidden,
            config.h_head_size(),
            &config.rwkv_state_readout_mode,
            config.recurrent_state_clamp,
            config.rwkv_channel_mix_key_clamp,
            config.rwkv_channel_mix_deepembed_clamp,
        )?;
        let h_to_context = take_linear(
            &mut map,
            "h_to_context",
            config.context_dim,
            config.h_hidden,
            true,
        )?;
        let h_halt_proj = take_linear(&mut map, "h_halt_proj", 1, config.h_hidden, true)?;
        let l_input_proj = take_linear(
            &mut map,
            "l_input_proj",
            config.l_hidden,
            config.context_dim * 2,
            true,
        )?;
        let l_rnn = RwkvCell::load(
            &mut map,
            "l_rnn",
            config.l_hidden,
            config.l_head_size(),
            &config.rwkv_state_readout_mode,
            config.recurrent_state_clamp,
            config.rwkv_channel_mix_key_clamp,
            config.rwkv_channel_mix_deepembed_clamp,
        )?;
        let context_drift_proj = take_linear(
            &mut map,
            "context_drift_proj",
            config.context_dim,
            config.l_hidden,
            false,
        )?;
        let l_to_out = take_linear(
            &mut map,
            "l_to_out",
            config.context_dim,
            config.l_hidden,
            true,
        )?;
        let out_norm = take_layer_norm(&mut map, "out_norm", config.context_dim)?;

        Ok(Self {
            config,
            token_embedding,
            persistent,
            memory_gate_warmup_step,
            ltm_gate_logit,
            rosa_gate_logit,
            ltm_router,
            rosa_router,
            deepembed,
            rosa_embedding,
            ltm,
            qproj,
            in_proj,
            l_feedback_proj,
            h_rnn,
            h_to_context,
            h_halt_proj,
            l_input_proj,
            l_rnn,
            context_drift_proj,
            l_to_out,
            out_norm,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn new_state(&self) -> InferenceState {
        let rosa_max = if self.config.use_rosa && self.config.enforce_rosa_max_context {
            self.config.rosa_max_context
        } else {
            0
        };
        InferenceState {
            h_state: self.h_rnn.initial_state(),
            l_state: self.l_rnn.initial_state(),
            prev_context: vec![0.0; self.config.context_dim],
            target_context: vec![0.0; self.config.context_dim],
            drift_state: vec![0.0; self.config.context_dim],
            global_pos: 0,
            rosa: if self.config.use_rosa {
                Some(RosaState::new(rosa_max))
            } else {
                None
            },
            ltm_fast_vals: None,
        }
    }

    pub fn prefill(&self, state: &mut InferenceState, tokens: &[usize]) -> Result<StepOutput> {
        if tokens.is_empty() {
            return Err(Error::InvalidConfig(
                "prefill requires at least one token".into(),
            ));
        }
        let mut result = None;
        for &token in tokens {
            result = Some(self.step(state, token)?);
        }
        Ok(result.expect("non-empty tokens"))
    }

    /// Consume one token and return the logits for the following token.
    ///
    /// The state object is intentionally separate from immutable model weights,
    /// allowing multiple independent conversations to share one loaded model.
    pub fn step(&self, state: &mut InferenceState, token: usize) -> Result<StepOutput> {
        if token >= self.config.vocab_size {
            return Err(Error::InvalidToken(token));
        }
        let raw_token = self.token_embedding.row(token).to_vec();
        let mut token_x = raw_token.clone();
        let gate_floor = self.memory_gate_floor();

        let rosa_prediction = if let Some(rosa) = &mut state.rosa {
            rosa.push(token)
        } else {
            None
        };
        if !matches!(self.rosa_embedding, RosaEmbedding::Off) {
            let mut rosa_features = self.rosa_features(rosa_prediction)?;
            let mut gate_logit = self.rosa_gate_logit.unwrap_or(0.0);
            if let Some(router) = &self.rosa_router {
                let mut routed = Vec::new();
                router.forward(&raw_token, &mut routed);
                gate_logit += routed[0];
            }
            let gate = apply_gate_floor(sigmoid(gate_logit.clamp(-50.0, 50.0)), gate_floor);
            for (x, feature) in token_x.iter_mut().zip(&mut rosa_features) {
                *x += gate * *feature;
            }
        }

        let (h_deep, l_deep) = self.deepembed_features(token, &raw_token)?;

        let mut q_input = Vec::with_capacity(self.config.context_dim * 2);
        q_input.extend_from_slice(&token_x);
        q_input.extend_from_slice(&state.prev_context);
        let mut query = Vec::new();
        self.qproj.forward(&q_input, &mut query);
        clamp_finite(&mut query, 12.0)?;
        let mut hits =
            self.ltm
                .retrieve_topk(&query, self.config.ltm_topk, state.ltm_fast_vals.as_deref())?;
        let topk_indices = hits.iter().map(|hit| hit.index).collect::<Vec<_>>();

        if self.config.ltm_time_feature_mode == "absolute-sinusoidal" {
            for hit in &mut hits {
                if hit.index.is_some() {
                    add_absolute_time_encoding(&mut hit.value, hit.timestamp);
                }
            }
        }
        let mut ltm_gate_logit = self.ltm_gate_logit;
        if let Some(router) = &self.ltm_router {
            let mut routed = Vec::new();
            router.forward(&token_x, &mut routed);
            ltm_gate_logit += routed[0];
        }
        let ltm_gate = apply_gate_floor(sigmoid(ltm_gate_logit.clamp(-50.0, 50.0)), gate_floor);

        let mut mac_input = Vec::with_capacity(
            self.config.context_dim
                + self.config.persistent_dim
                + self.config.ltm_topk * self.config.ltm_val_dim,
        );
        mac_input.extend_from_slice(&token_x);
        mac_input.extend_from_slice(&self.persistent);
        for hit in &hits {
            mac_input.extend(hit.value.iter().map(|v| v * ltm_gate));
        }
        let mut enc = Vec::new();
        self.in_proj.forward(&mac_input, &mut enc);
        for value in &mut enc {
            *value = gelu(*value);
        }
        clamp_finite(&mut enc, 30.0)?;

        let mut l_feedback = Vec::new();
        self.l_feedback_proj
            .forward(self.l_rnn.state_hidden(&state.l_state), &mut l_feedback);
        let mut enc_with_feedback = enc
            .iter()
            .zip(&l_feedback)
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>();
        clamp_finite(&mut enc_with_feedback, self.config.activation_clamp)?;

        let (h_out_real, real_h_state) =
            self.h_rnn
                .forward(&enc_with_feedback, &state.h_state, h_deep.as_deref())?;
        state.h_state = real_h_state;

        if state.global_pos % self.config.h_stride == 0 {
            state.prev_context = state.target_context.clone();
            let (manager_output, manager_state) = self.manager_plan(
                &enc_with_feedback,
                h_out_real,
                state.h_state.clone(),
                h_deep.as_deref(),
            )?;
            state.h_state = manager_state;
            self.h_to_context
                .forward(&manager_output, &mut state.target_context);
            clamp_finite(&mut state.target_context, self.config.context_state_clamp)?;
        }

        let alpha = (state.global_pos % self.config.h_stride) as f32 / self.config.h_stride as f32;
        let mut sliding_context = vec![0.0; self.config.context_dim];
        for i in 0..self.config.context_dim {
            sliding_context[i] =
                state.prev_context[i] + alpha * (state.target_context[i] - state.prev_context[i]);
        }
        clamp_finite(&mut sliding_context, self.config.context_state_clamp)?;

        let exact_full_sample = self.config.full_sample_bptt;
        let aligned_legacy_boundary = !exact_full_sample
            && self.config.drift_recurrence_mode == "legacy-chunk-seeded"
            && self.config.training_chunk_size > 0
            && state.global_pos > 0
            && state.global_pos % self.config.training_chunk_size == 0;
        let mut initial_drift = if aligned_legacy_boundary {
            state.drift_state.clone()
        } else {
            let mut derived = Vec::new();
            self.context_drift_proj
                .forward(self.l_rnn.state_hidden(&state.l_state), &mut derived);
            for value in &mut derived {
                *value = value.tanh();
            }
            derived
        };
        clamp_finite(&mut initial_drift, self.config.drift_state_clamp)?;
        l2_norm_clamp(&mut initial_drift, self.config.drift_norm_clamp)?;

        let (mut final_enc, next_l_state, final_drift) = self.worker_step(
            &enc,
            &sliding_context,
            &state.l_state,
            initial_drift,
            l_deep.as_deref(),
        )?;
        state.l_state = next_l_state;
        state.drift_state = final_drift;
        clamp_finite(&mut final_enc, self.config.activation_clamp)?;

        let mut normalized = Vec::new();
        self.out_norm.forward(&final_enc, &mut normalized);
        clamp_finite(&mut normalized, self.config.activation_clamp)?;
        let mut logits = Vec::new();
        self.token_embedding.linear(&normalized, &mut logits);
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("language-model logits".into()));
        }
        if self.config.inference_logit_clamp > 0.0 {
            let max = self.config.inference_logit_clamp;
            for value in &mut logits {
                *value = value.clamp(-max, max);
            }
        }

        state.global_pos += 1;
        Ok(StepOutput {
            logits,
            topk_indices,
        })
    }

    pub fn generate(
        &self,
        state: &mut InferenceState,
        prompt: &[usize],
        max_new_tokens: usize,
        eos_token: Option<usize>,
        generation: &GenerationConfig,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(Error::InvalidConfig(
                "generation prompt cannot be empty".into(),
            ));
        }
        if generation.repetition_penalty <= 0.0 {
            return Err(Error::InvalidConfig(
                "repetition_penalty must be greater than zero".into(),
            ));
        }
        if !(0.0 < generation.top_p && generation.top_p <= 1.0) {
            return Err(Error::InvalidConfig("top_p must be in (0, 1]".into()));
        }

        let mut history = prompt.to_vec();
        let mut output = prompt.to_vec();
        let mut step = self.prefill(state, prompt)?;
        let mut rng = XorShift64::new(generation.seed);
        for _ in 0..max_new_tokens {
            let token = sample_token(&step.logits, &history, generation, &mut rng)?;
            output.push(token);
            history.push(token);
            if eos_token == Some(token) {
                break;
            }
            step = self.step(state, token)?;
        }
        Ok(output)
    }

    fn memory_gate_floor(&self) -> Option<f32> {
        if self.config.memory_gate_warmup_steps == 0 || self.config.memory_gate_warmup_floor <= 0.0
        {
            return None;
        }
        let progress = (self.memory_gate_warmup_step / self.config.memory_gate_warmup_steps as f32)
            .clamp(0.0, 1.0);
        Some(self.config.memory_gate_warmup_floor.clamp(0.0, 0.95) * (1.0 - progress))
    }

    fn rosa_features(&self, prediction: Option<usize>) -> Result<Vec<f32>> {
        match &self.rosa_embedding {
            RosaEmbedding::Off => Ok(vec![0.0; self.config.context_dim]),
            RosaEmbedding::Legacy(table) => {
                let id = prediction
                    .filter(|&id| id < self.config.vocab_size)
                    .unwrap_or(self.config.vocab_size);
                if id == self.config.vocab_size && self.config.rosa_zero_no_prediction {
                    Ok(vec![0.0; self.config.context_dim])
                } else {
                    Ok(table.row(id).to_vec())
                }
            }
            RosaEmbedding::Shared(adapter) => match prediction {
                Some(id) if id < self.config.vocab_size => {
                    adapter.forward(self.token_embedding.row(id))
                }
                _ => Ok(vec![0.0; self.config.context_dim]),
            },
        }
    }

    fn deepembed_features(
        &self,
        token: usize,
        raw_token: &[f32],
    ) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>)> {
        match &self.deepembed {
            DeepEmbed::Off => Ok((None, None)),
            DeepEmbed::Legacy { h_table, l_table } => Ok((
                Some(h_table.row(token).to_vec()),
                Some(l_table.row(token).to_vec()),
            )),
            DeepEmbed::Shared {
                h_adapter,
                l_adapter,
            } => {
                let mut normalized = Vec::new();
                layer_norm_no_affine(raw_token, LAYER_NORM_EPS, &mut normalized);
                Ok((
                    Some(h_adapter.forward_normalized(&normalized)?),
                    Some(l_adapter.forward_normalized(&normalized)?),
                ))
            }
        }
    }

    fn halt_probability(&self, h_out: &[f32]) -> Result<f32> {
        let mut logit = Vec::new();
        self.h_halt_proj.forward(h_out, &mut logit);
        if !logit[0].is_finite() {
            return Err(Error::NonFinite("manager halt logit".into()));
        }
        Ok(
            sigmoid(logit[0].clamp(-self.config.halt_logit_clamp, self.config.halt_logit_clamp))
                .clamp(1.0e-6, 1.0 - 1.0e-6),
        )
    }

    fn manager_plan(
        &self,
        manager_input: &[f32],
        first_output: Vec<f32>,
        first_state: RwkvState,
        deepembed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, RwkvState)> {
        let mut outputs = vec![first_output];
        let mut states = vec![first_state];
        let mut probabilities = vec![self.halt_probability(&outputs[0])?];

        if self.config.manager_compute_mode == "hard-masked" {
            let mut survival = 1.0 - probabilities[0];
            for step_index in 1..self.config.max_h_steps {
                let completed = step_index;
                let cumulative = 1.0 - survival;
                if completed >= self.config.min_h_steps && cumulative >= self.config.h_halt_thresh {
                    break;
                }
                let (out, next) = self.h_rnn.forward(
                    manager_input,
                    states.last().expect("manager state"),
                    deepembed,
                )?;
                let p = self.halt_probability(&out)?;
                outputs.push(out);
                states.push(next);
                probabilities.push(p);
                survival *= 1.0 - p;
            }
            // Pick the first cumulative CDF crossing after min_h_steps, or the
            // last computed state if none crossed.
            let mut survival = 1.0f32;
            let mut selected = outputs.len() - 1;
            for (index, &p) in probabilities.iter().enumerate() {
                survival *= 1.0 - p;
                let completed = index + 1;
                if completed >= self.config.min_h_steps
                    && (1.0 - survival) >= self.config.h_halt_thresh
                {
                    selected = index;
                    break;
                }
            }
            return Ok((outputs[selected].clone(), states[selected].clone()));
        }

        for _ in 1..self.config.max_h_steps {
            if !self.config.inference_logit_parity
                && probabilities.last().copied().unwrap_or(0.0) > self.config.h_halt_thresh
            {
                break;
            }
            let (out, next) = self.h_rnn.forward(
                manager_input,
                states.last().expect("manager state"),
                deepembed,
            )?;
            probabilities.push(self.halt_probability(&out)?);
            outputs.push(out);
            states.push(next);
        }
        let (weights, remainder) = normalized_act_weights(&probabilities);
        let mut manager_output = vec![0.0f32; self.config.h_hidden];
        for (step, output) in outputs.iter().enumerate() {
            let mut weight = weights[step];
            if step + 1 == outputs.len() {
                weight += remainder;
            }
            for i in 0..manager_output.len() {
                manager_output[i] += weight * output[i];
            }
        }
        let manager_state = match self.config.manager_state_commit_mode.as_str() {
            "legacy-real-step" => states[0].clone(),
            "last-shadow" => states.last().expect("manager state").clone(),
            "act-weighted" => blend_states(&states, &weights, remainder),
            "hard-selected" => {
                return Err(Error::InvalidConfig(
                    "hard-selected manager state requires hard-masked compute".into(),
                ))
            }
            other => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported manager_state_commit_mode {other:?}"
                )))
            }
        };
        Ok((manager_output, manager_state))
    }

    fn worker_step(
        &self,
        enc: &[f32],
        static_context: &[f32],
        original_l_state: &RwkvState,
        mut current_drift: Vec<f32>,
        deepembed: Option<&[f32]>,
    ) -> Result<(Vec<f32>, RwkvState, Vec<f32>)> {
        clamp_finite(&mut current_drift, self.config.drift_state_clamp)?;
        l2_norm_clamp(&mut current_drift, self.config.drift_norm_clamp)?;
        let mut shadow = original_l_state.clone();
        let mut l_input = self.worker_input(enc, static_context, &current_drift)?;
        let training_parity = self.config.inference_logit_parity || self.config.full_sample_bptt;

        if training_parity {
            for _ in 0..self.config.max_l_steps {
                let (mut l_out, candidate_state) =
                    self.l_rnn.forward(&l_input, &shadow, deepembed)?;
                clamp_finite(&mut l_out, self.config.activation_clamp)?;
                let drift_delta = self.drift_delta(&l_out)?;
                let mut candidate_drift = current_drift
                    .iter()
                    .zip(&drift_delta)
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>();
                clamp_finite(&mut candidate_drift, self.config.drift_state_clamp)?;
                l2_norm_clamp(&mut candidate_drift, self.config.drift_norm_clamp)?;
                let candidate_input = self.worker_input(enc, static_context, &candidate_drift)?;
                shadow = candidate_state;
                current_drift = candidate_drift;
                l_input = candidate_input;
                let mean_abs =
                    drift_delta.iter().map(|v| v.abs()).sum::<f32>() / drift_delta.len() as f32;
                if mean_abs < self.config.l_conv_atol {
                    break;
                }
            }
        } else {
            let mut previous_shadow = shadow.clone();
            for _ in 0..self.config.max_l_steps {
                let (mut l_out, candidate_state) =
                    self.l_rnn.forward(&l_input, &shadow, deepembed)?;
                clamp_finite(&mut l_out, self.config.activation_clamp)?;
                let drift_delta = self.drift_delta(&l_out)?;
                let mut candidate_drift = current_drift
                    .iter()
                    .zip(&drift_delta)
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>();
                clamp_finite(&mut candidate_drift, self.config.drift_state_clamp)?;
                l2_norm_clamp(&mut candidate_drift, self.config.drift_norm_clamp)?;
                let candidate_input = self.worker_input(enc, static_context, &candidate_drift)?;
                let drift_converged = drift_delta.iter().map(|v| v.abs()).sum::<f32>()
                    / (drift_delta.len() as f32)
                    < self.config.l_conv_atol;
                let state_converged = legacy_state_allclose(
                    &candidate_state,
                    &previous_shadow,
                    self.config.l_conv_atol,
                );
                shadow = candidate_state;
                current_drift = candidate_drift;
                l_input = candidate_input;
                if drift_converged || state_converged {
                    break;
                }
                previous_shadow = shadow.clone();
            }
        }

        let (mut final_l_out, next_l_state) =
            self.l_rnn.forward(&l_input, original_l_state, deepembed)?;
        clamp_finite(&mut final_l_out, self.config.activation_clamp)?;
        let mut projected = Vec::new();
        self.l_to_out.forward(&final_l_out, &mut projected);
        let mut final_enc = enc
            .iter()
            .zip(&projected)
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>();
        clamp_finite(&mut final_enc, self.config.activation_clamp)?;
        Ok((final_enc, next_l_state, current_drift))
    }

    fn worker_input(&self, enc: &[f32], static_context: &[f32], drift: &[f32]) -> Result<Vec<f32>> {
        let mut input = Vec::with_capacity(self.config.context_dim * 2);
        input.extend_from_slice(enc);
        input.extend(
            static_context
                .iter()
                .zip(drift)
                .map(|(&context, &delta)| context + delta),
        );
        let mut projected = Vec::new();
        self.l_input_proj.forward(&input, &mut projected);
        clamp_finite(&mut projected, self.config.recurrent_state_clamp)?;
        Ok(projected)
    }

    fn drift_delta(&self, l_out: &[f32]) -> Result<Vec<f32>> {
        let mut delta = Vec::new();
        self.context_drift_proj.forward(l_out, &mut delta);
        for value in &mut delta {
            *value = value.tanh() * self.config.drift_delta_scale;
        }
        if delta.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("worker drift update".into()));
        }
        Ok(delta)
    }
}

fn apply_gate_floor(gate: f32, floor: Option<f32>) -> f32 {
    match floor {
        Some(floor) => floor + (1.0 - floor) * gate,
        None => gate,
    }
}

fn add_absolute_time_encoding(values: &mut [f32], timestamp: f32) {
    let half = values.len() / 2;
    if half == 0 {
        return;
    }
    for i in 0..half {
        let freq = if half == 1 {
            1.0
        } else {
            (-(i as f32) * 10000.0f32.ln() / (half - 1) as f32).exp()
        };
        let arg = timestamp * freq;
        values[i] += arg.sin();
        values[half + i] += arg.cos();
    }
    // Odd value widths receive one trailing zero in Python, so no change.
}

fn normalized_act_weights(probabilities: &[f32]) -> (Vec<f32>, f32) {
    let mut survival = 1.0f32;
    let mut weights = Vec::with_capacity(probabilities.len());
    for &probability in probabilities {
        let p = probability.clamp(1.0e-6, 1.0 - 1.0e-6);
        weights.push(p * survival);
        survival *= 1.0 - p;
    }
    let remainder = survival;
    let total = (weights.iter().sum::<f32>() + remainder).max(1.0e-8);
    for weight in &mut weights {
        *weight /= total;
    }
    (weights, remainder / total)
}

fn blend_states(states: &[RwkvState], weights: &[f32], remainder: f32) -> RwkvState {
    debug_assert!(!states.is_empty());
    let mut output = RwkvState {
        prev_tm: vec![0.0; states[0].prev_tm.len()],
        prev_cm: vec![0.0; states[0].prev_cm.len()],
        v_first: vec![0.0; states[0].v_first.len()],
        output: vec![0.0; states[0].output.len()],
        matrix: vec![0.0; states[0].matrix.len()],
    };
    for (step, state) in states.iter().enumerate() {
        let mut weight = weights[step];
        if step + 1 == states.len() {
            weight += remainder;
        }
        accumulate(&mut output.prev_tm, &state.prev_tm, weight);
        accumulate(&mut output.prev_cm, &state.prev_cm, weight);
        accumulate(&mut output.v_first, &state.v_first, weight);
        accumulate(&mut output.output, &state.output, weight);
        accumulate(&mut output.matrix, &state.matrix, weight);
    }
    output
}

fn accumulate(out: &mut [f32], value: &[f32], weight: f32) {
    for (target, &source) in out.iter_mut().zip(value) {
        *target += weight * source;
    }
}

fn legacy_state_allclose(a: &RwkvState, b: &RwkvState, atol: f32) -> bool {
    allclose(&a.prev_tm, &b.prev_tm, atol)
        && allclose(&a.prev_cm, &b.prev_cm, atol)
        && allclose(&a.v_first, &b.v_first, atol)
}

fn allclose(a: &[f32], b: &[f32], atol: f32) -> bool {
    a.iter()
        .zip(b)
        .all(|(&left, &right)| (left - right).abs() <= atol + 1.0e-5 * right.abs())
}

fn sample_token(
    logits: &[f32],
    history: &[usize],
    config: &GenerationConfig,
    rng: &mut XorShift64,
) -> Result<usize> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite("sampling logits".into()));
    }
    let penalized: HashSet<usize> = if config.repetition_penalty != 1.0 {
        history
            .iter()
            .copied()
            .filter(|&id| id < logits.len())
            .collect()
    } else {
        HashSet::new()
    };
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(id, mut score)| {
            if penalized.contains(&id) {
                score = if score > 0.0 {
                    score / config.repetition_penalty
                } else {
                    score * config.repetition_penalty
                };
            }
            (id, score)
        })
        .collect::<Vec<_>>();
    if config.temperature <= 0.0 {
        return Ok(candidates
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty logits")
            .0);
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }
    let inv_temperature = 1.0 / config.temperature.max(1.0e-6);
    let mut scores = candidates
        .iter()
        .map(|(_, score)| score * inv_temperature)
        .collect::<Vec<_>>();
    softmax_in_place(&mut scores)?;

    if config.top_p < 1.0 {
        let mut cumulative_before = 0.0f32;
        let mut keep = scores.len();
        for (index, &probability) in scores.iter().enumerate() {
            if index > 0 && cumulative_before > config.top_p {
                keep = index;
                break;
            }
            cumulative_before += probability;
        }
        candidates.truncate(keep);
        scores.truncate(keep);
        let sum = scores.iter().sum::<f32>();
        for probability in &mut scores {
            *probability /= sum;
        }
    }

    let draw = rng.next_f32();
    let mut cumulative = 0.0f32;
    for ((token, _), probability) in candidates.iter().zip(&scores) {
        cumulative += *probability;
        if draw <= cumulative {
            return Ok(*token);
        }
    }
    Ok(candidates.last().expect("sampling candidates").0)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_weights_preserve_mass() {
        let (weights, remainder) = normalized_act_weights(&[0.2, 0.3, 0.4]);
        let total = weights.iter().sum::<f32>() + remainder;
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn greedy_sampling_returns_argmax() {
        let config = GenerationConfig {
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        let mut rng = XorShift64::new(1);
        assert_eq!(
            sample_token(&[1.0, 4.0, 2.0], &[], &config, &mut rng).unwrap(),
            1
        );
    }
}
