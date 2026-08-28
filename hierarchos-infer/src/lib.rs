//! Pure Rust, f32-only inference runtime for Hierarchos.
//!
//! The runtime deliberately does not depend on Python, PyTorch, BLAS, or a C/C++
//! tensor library. Model conversion is an offline step; once a `.hrf32` file has
//! been produced, loading and inference are Rust-only.

mod config;
mod error;
mod format;
mod ltm;
mod math;
mod model;
mod rosa;
mod rwkv;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use format::{ModelFile, Tensor};
pub use model::{GenerationConfig, Hierarchos, InferenceState, StepOutput};
