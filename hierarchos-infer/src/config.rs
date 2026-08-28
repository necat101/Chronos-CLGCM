use serde::Deserialize;

use crate::{Error, Result};

fn default_true() -> bool {
    true
}
fn default_persistent_dim() -> usize {
    128
}
fn default_ltm_slots() -> usize {
    1024
}
fn default_ltm_dim() -> usize {
    128
}
fn default_ltm_topk() -> usize {
    4
}
fn default_h_stride() -> usize {
    4
}
fn default_max_steps() -> usize {
    5
}
fn default_min_h_steps() -> usize {
    1
}
fn default_halt_threshold() -> f32 {
    0.9
}
fn default_l_conv_atol() -> f32 {
    0.01
}
fn default_state_clamp() -> f32 {
    50.0
}
fn default_context_clamp() -> f32 {
    50.0
}
fn default_drift_clamp() -> f32 {
    5.0
}
fn default_activation_clamp() -> f32 {
    100.0
}
fn default_halt_clamp() -> f32 {
    30.0
}
fn default_cm_key_clamp() -> f32 {
    12.0
}
fn default_cm_deep_clamp() -> f32 {
    4.0
}
fn default_drift_scale() -> f32 {
    1.0
}
fn default_inference_logit_clamp() -> f32 {
    30.0
}
fn default_rosa_context() -> usize {
    512
}
fn default_training_chunk_size() -> usize {
    128
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub context_dim: usize,
    #[serde(default = "default_persistent_dim")]
    pub persistent_dim: usize,
    #[serde(default = "default_ltm_slots")]
    pub ltm_slots: usize,
    #[serde(default = "default_ltm_dim")]
    pub ltm_key_dim: usize,
    #[serde(default = "default_ltm_dim")]
    pub ltm_val_dim: usize,
    #[serde(default = "default_ltm_topk")]
    pub ltm_topk: usize,
    pub h_hidden: usize,
    pub l_hidden: usize,
    #[serde(default = "default_h_stride")]
    pub h_stride: usize,
    #[serde(default = "default_max_steps")]
    pub max_h_steps: usize,
    #[serde(default = "default_max_steps")]
    pub max_l_steps: usize,
    #[serde(default = "default_min_h_steps")]
    pub min_h_steps: usize,
    #[serde(default = "default_halt_threshold")]
    pub h_halt_thresh: f32,
    #[serde(default = "default_l_conv_atol")]
    pub l_conv_atol: f32,
    #[serde(default)]
    pub drift_norm_clamp: f32,
    #[serde(default = "default_drift_scale")]
    pub drift_delta_scale: f32,
    #[serde(default = "default_state_clamp")]
    pub recurrent_state_clamp: f32,
    #[serde(default = "default_context_clamp")]
    pub context_state_clamp: f32,
    #[serde(default = "default_drift_clamp")]
    pub drift_state_clamp: f32,
    #[serde(default = "default_activation_clamp")]
    pub activation_clamp: f32,
    #[serde(default = "default_halt_clamp")]
    pub halt_logit_clamp: f32,
    #[serde(default = "default_cm_key_clamp")]
    pub rwkv_channel_mix_key_clamp: f32,
    #[serde(default = "default_cm_deep_clamp")]
    pub rwkv_channel_mix_deepembed_clamp: f32,
    #[serde(default)]
    pub h_rwkv_head_size: Option<usize>,
    #[serde(default)]
    pub l_rwkv_head_size: Option<usize>,
    #[serde(default)]
    pub rwkv_head_size: Option<usize>,
    #[serde(default = "default_true")]
    pub use_deepembed: bool,
    #[serde(default = "default_true")]
    pub use_rosa: bool,
    #[serde(default = "default_true")]
    pub memory_token_routers: bool,
    #[serde(default = "default_rosa_context")]
    pub rosa_max_context: usize,
    #[serde(default)]
    pub enforce_rosa_max_context: bool,
    #[serde(default)]
    pub rosa_zero_no_prediction: bool,
    #[serde(default)]
    pub token_adapter_rank: Option<usize>,
    #[serde(default = "default_training_chunk_size")]
    pub training_chunk_size: usize,
    #[serde(default = "default_inference_logit_clamp")]
    pub inference_logit_clamp: f32,
    #[serde(default)]
    pub inference_logit_parity: bool,
    #[serde(default)]
    pub full_sample_bptt: bool,
    #[serde(default)]
    pub memory_gate_warmup_steps: usize,
    #[serde(default)]
    pub memory_gate_warmup_floor: f32,
    #[serde(default = "default_legacy_revision")]
    pub architecture_revision: String,
    #[serde(default = "default_legacy_deepembed")]
    pub deepembed_mode: String,
    #[serde(default = "default_legacy_rosa")]
    pub rosa_embedding_mode: String,
    #[serde(default = "default_legacy_readout")]
    pub rwkv_state_readout_mode: String,
    #[serde(default = "default_legacy_drift")]
    pub drift_recurrence_mode: String,
    #[serde(default = "default_soft_act")]
    pub manager_compute_mode: String,
    #[serde(default = "default_legacy_manager_commit")]
    pub manager_state_commit_mode: String,
    #[serde(default = "default_absolute_time")]
    pub ltm_time_feature_mode: String,
}

fn default_legacy_revision() -> String {
    "legacy-v8".into()
}
fn default_legacy_deepembed() -> String {
    "legacy-table".into()
}
fn default_legacy_rosa() -> String {
    "legacy-table".into()
}
fn default_legacy_readout() -> String {
    "legacy-input-cache".into()
}
fn default_legacy_drift() -> String {
    "legacy-chunk-seeded".into()
}
fn default_soft_act() -> String {
    "soft-act".into()
}
fn default_legacy_manager_commit() -> String {
    "legacy-real-step".into()
}
fn default_absolute_time() -> String {
    "absolute-sinusoidal".into()
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("context_dim", self.context_dim),
            ("persistent_dim", self.persistent_dim),
            ("ltm_slots", self.ltm_slots),
            ("ltm_key_dim", self.ltm_key_dim),
            ("ltm_val_dim", self.ltm_val_dim),
            ("ltm_topk", self.ltm_topk),
            ("h_hidden", self.h_hidden),
            ("l_hidden", self.l_hidden),
            ("h_stride", self.h_stride),
            ("max_h_steps", self.max_h_steps),
            ("max_l_steps", self.max_l_steps),
            ("min_h_steps", self.min_h_steps),
        ] {
            if value == 0 {
                return Err(Error::InvalidConfig(format!("{name} must be positive")));
            }
        }
        if self.min_h_steps > self.max_h_steps {
            return Err(Error::InvalidConfig(
                "min_h_steps cannot exceed max_h_steps".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.h_halt_thresh) {
            return Err(Error::InvalidConfig(
                "h_halt_thresh must be in [0, 1]".into(),
            ));
        }
        if self.deepembed_mode != "off"
            && self.deepembed_mode != "legacy-table"
            && self.deepembed_mode != "shared-factorized"
        {
            return Err(Error::InvalidConfig(format!(
                "unsupported deepembed_mode {:?}",
                self.deepembed_mode
            )));
        }
        if self.rosa_embedding_mode != "off"
            && self.rosa_embedding_mode != "legacy-table"
            && self.rosa_embedding_mode != "shared-factorized"
        {
            return Err(Error::InvalidConfig(format!(
                "unsupported rosa_embedding_mode {:?}",
                self.rosa_embedding_mode
            )));
        }
        if self.rwkv_state_readout_mode != "legacy-input-cache"
            && self.rwkv_state_readout_mode != "explicit-output"
        {
            return Err(Error::InvalidConfig(format!(
                "unsupported rwkv_state_readout_mode {:?}",
                self.rwkv_state_readout_mode
            )));
        }
        if self.manager_compute_mode != "soft-act" && self.manager_compute_mode != "hard-masked" {
            return Err(Error::InvalidConfig(format!(
                "unsupported manager_compute_mode {:?}",
                self.manager_compute_mode
            )));
        }
        Ok(())
    }

    pub(crate) fn h_head_size(&self) -> Option<usize> {
        self.h_rwkv_head_size.or(self.rwkv_head_size)
    }

    pub(crate) fn l_head_size(&self) -> Option<usize> {
        self.l_rwkv_head_size.or(self.rwkv_head_size)
    }
}
