use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::checkpoint::replace_f32_tensor_values;

pub const HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME: &str = "adapter_config.json";
pub const HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME: &str = "adapter_model.safetensors";
pub const HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME: &str = "hierarchos_adapter_manifest.json";

const HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT: &str = "hierarchos-peft-lora-v1";
const HIERARCHOS_LORA_ADAPTER_MANIFEST_VERSION: u64 = 1;
const PEFT_STATE_PREFIX: &str = "base_model.model.";
const LORA_A_SUFFIX: &str = ".lora_A.weight";
const LORA_B_SUFFIX: &str = ".lora_B.weight";

#[derive(Clone, Debug, PartialEq)]
pub struct HierarchosNativeLoraMergeReport {
    pub merged_lora_modules: usize,
    pub replaced_module_tensors: usize,
    pub base_checkpoint_sha256: String,
    pub adapter_checkpoint_sha256: String,
    pub architecture_contract_sha256: String,
}

#[derive(Debug, Deserialize)]
struct LoraAdapterConfig {
    #[serde(default)]
    peft_type: String,
    #[serde(default)]
    task_type: String,
    r: usize,
    lora_alpha: f64,
    #[serde(default)]
    lora_dropout: f64,
    target_modules: Vec<String>,
    #[serde(default)]
    modules_to_save: Vec<String>,
    #[serde(default)]
    bias: String,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    use_qalora: bool,
    #[serde(default)]
    lora_bias: bool,
    #[serde(default)]
    target_parameters: Option<serde_json::Value>,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
    #[serde(default)]
    alora_invocation_tokens: Option<serde_json::Value>,
    #[serde(default)]
    arrow_config: Option<serde_json::Value>,
    #[serde(default)]
    corda_config: Option<serde_json::Value>,
    #[serde(default)]
    rank_pattern: BTreeMap<String, usize>,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct ManifestFileBinding {
    filename: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct LoraAdapterManifest {
    manifest_version: u64,
    format: String,
    base_checkpoint: ManifestFileBinding,
    architecture_contract_sha256: String,
    adapter_files: BTreeMap<String, String>,
}

#[derive(Debug)]
struct AdapterPair {
    module_name: String,
    a_key: String,
    b_key: String,
}

/// Merge the exact SafeTensors PEFT-LoRA package emitted by Hierarchos into a
/// standalone canonical model checkpoint without importing Python, PEFT, or
/// PyTorch. The model tensor layout remains the shared row-major ABI consumed
/// by native Rust inference, the Vulkan trainer, and compatible CUDA readers.
///
/// This is deliberately a package/checkpoint operation rather than a training
/// primitive. Training remains Vulkan-only; the small A/B merge is deterministic
/// native Rust arithmetic and is performed once when publishing an adapter.
pub fn merge_hierarchos_lora_safetensors(
    base_weights: &Path,
    adapter_dir: &Path,
    destination: &Path,
) -> Result<HierarchosNativeLoraMergeReport> {
    if !base_weights.is_file() {
        bail!(
            "base model weights do not exist: {}",
            base_weights.display()
        );
    }
    if !adapter_dir.is_dir() {
        bail!(
            "LoRA adapter directory does not exist: {}",
            adapter_dir.display()
        );
    }
    if base_weights == destination {
        bail!("native LoRA merge requires a distinct destination checkpoint");
    }

    reject_pickle_adapter_files(adapter_dir)?;
    let config_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME);
    let adapter_weights = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME);
    let manifest_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME);
    let config = read_adapter_config(&config_path)?;
    validate_adapter_config(&config)?;
    let base_sha256 = sha256_file(base_weights)?;
    let adapter_sha256 = sha256_file(&adapter_weights)?;
    let architecture_contract_sha256 = validate_bound_manifest(
        base_weights,
        &config_path,
        &adapter_weights,
        &manifest_path,
        &base_sha256,
    )?;

    let base_bytes = fs::read(base_weights)
        .with_context(|| format!("reading base model {}", base_weights.display()))?;
    let base_tensors = SafeTensors::deserialize(&base_bytes)
        .with_context(|| format!("parsing base model {}", base_weights.display()))?;
    let adapter_bytes = fs::read(&adapter_weights)
        .with_context(|| format!("reading adapter {}", adapter_weights.display()))?;
    let adapter_tensors = SafeTensors::deserialize(&adapter_bytes)
        .with_context(|| format!("parsing adapter {}", adapter_weights.display()))?;

    let (pairs, saved_module_keys) = discover_adapter_tensors(&adapter_tensors, &config)?;
    let mut replacements = BTreeMap::<String, Vec<f32>>::new();

    for pair in &pairs {
        let a = adapter_tensors.tensor(&pair.a_key)?;
        let b = adapter_tensors.tensor(&pair.b_key)?;
        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            bail!(
                "LoRA module {:?} requires rank-2 A/B tensors; got A={a_shape:?} B={b_shape:?}",
                pair.module_name
            );
        }
        let rank = a_shape[0];
        if rank == 0 || b_shape[1] != rank {
            bail!(
                "LoRA module {:?} has inconsistent rank geometry A={a_shape:?} B={b_shape:?}",
                pair.module_name
            );
        }
        let expected_rank = pattern_value(&pair.module_name, &config.rank_pattern)
            .copied()
            .unwrap_or(config.r);
        if rank != expected_rank {
            bail!(
                "LoRA module {:?} has rank {rank}, but adapter config requires {expected_rank}",
                pair.module_name
            );
        }
        let alpha = pattern_value(&pair.module_name, &config.alpha_pattern)
            .copied()
            .unwrap_or(config.lora_alpha);
        if !alpha.is_finite() || alpha <= 0.0 {
            bail!(
                "LoRA module {:?} has invalid alpha {alpha}",
                pair.module_name
            );
        }
        let scale = if config.use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        } as f32;
        if !scale.is_finite() {
            bail!(
                "LoRA module {:?} produced non-finite scaling",
                pair.module_name
            );
        }

        let target_name = format!("{}.weight", pair.module_name);
        let base = base_tensors.tensor(&target_name).with_context(|| {
            format!(
                "adapter targets tensor {target_name:?}, which is absent from {}",
                base_weights.display()
            )
        })?;
        let base_shape = base.shape().to_vec();
        let a_values = decode_float_tensor(&pair.a_key, &a)?;
        let b_values = decode_float_tensor(&pair.b_key, &b)?;
        let mut merged = decode_float_tensor(&target_name, &base)?;

        let out_dim = b_shape[0];
        let in_dim = a_shape[1];
        let expected_shape = if config.fan_in_fan_out {
            vec![in_dim, out_dim]
        } else {
            vec![out_dim, in_dim]
        };
        if base_shape != expected_shape {
            bail!(
                "LoRA module {:?} resolves to base shape {base_shape:?}; expected {expected_shape:?} from A={a_shape:?} B={b_shape:?}",
                pair.module_name
            );
        }
        add_lora_delta(
            &mut merged,
            &a_values,
            &b_values,
            out_dim,
            rank,
            in_dim,
            scale,
            config.fan_in_fan_out,
        )?;
        if replacements.insert(target_name.clone(), merged).is_some() {
            bail!("adapter attempts to replace base tensor {target_name:?} twice");
        }
    }

    for saved_key in &saved_module_keys {
        let target_name = saved_key
            .strip_prefix(PEFT_STATE_PREFIX)
            .context("validated modules_to_save tensor lost PEFT state prefix")?;
        let saved = adapter_tensors.tensor(saved_key)?;
        let base = base_tensors.tensor(target_name).with_context(|| {
            format!(
                "adapter modules_to_save tensor {saved_key:?} resolves to missing base tensor {target_name:?}"
            )
        })?;
        if saved.shape() != base.shape() {
            bail!(
                "adapter modules_to_save tensor {saved_key:?} has shape {:?}; base tensor {target_name:?} has shape {:?}",
                saved.shape(),
                base.shape()
            );
        }
        let values = decode_float_tensor(saved_key, &saved)?;
        if replacements
            .insert(target_name.to_string(), values)
            .is_some()
        {
            bail!("adapter attempts to replace base tensor {target_name:?} twice");
        }
    }

    if replacements.is_empty() {
        bail!("LoRA adapter contains no mergeable tensors");
    }
    drop(adapter_tensors);
    drop(adapter_bytes);
    drop(base_tensors);
    drop(base_bytes);

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating merge output directory {}", parent.display()))?;
    }
    let replacement_refs = replacements
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect::<Vec<_>>();
    replace_f32_tensor_values(base_weights, destination, &replacement_refs)
        .with_context(|| format!("writing merged model {}", destination.display()))?;

    Ok(HierarchosNativeLoraMergeReport {
        merged_lora_modules: pairs.len(),
        replaced_module_tensors: saved_module_keys.len(),
        base_checkpoint_sha256: base_sha256,
        adapter_checkpoint_sha256: adapter_sha256,
        architecture_contract_sha256,
    })
}

fn read_adapter_config(path: &Path) -> Result<LoraAdapterConfig> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn validate_adapter_config(config: &LoraAdapterConfig) -> Result<()> {
    if !config.peft_type.eq_ignore_ascii_case("LORA") {
        bail!(
            "unsupported PEFT adapter type {:?}; expected LORA",
            config.peft_type
        );
    }
    if !config.task_type.eq_ignore_ascii_case("CAUSAL_LM") {
        bail!(
            "unsupported PEFT task type {:?}; Hierarchos LoRA requires CAUSAL_LM",
            config.task_type
        );
    }
    if config.r == 0 || !config.lora_alpha.is_finite() || config.lora_alpha <= 0.0 {
        bail!("LoRA rank and alpha must both be finite and positive");
    }
    if !config.lora_dropout.is_finite() || config.lora_dropout < 0.0 || config.lora_dropout >= 1.0 {
        bail!("LoRA dropout must be finite and in [0, 1)");
    }
    if !config.bias.is_empty() && !config.bias.eq_ignore_ascii_case("none") {
        bail!("native Hierarchos merge supports PEFT bias='none' only");
    }
    if config.target_modules.is_empty() {
        bail!("PEFT LoRA adapter has no target_modules");
    }
    let modules_to_save = config
        .modules_to_save
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if modules_to_save.iter().any(|module| *module != "ltm") {
        bail!("native Hierarchos merge only supports the optional 'ltm' modules_to_save entry");
    }
    let unsupported = [
        ("use_dora", config.use_dora),
        ("use_qalora", config.use_qalora),
        ("lora_bias", config.lora_bias),
        (
            "target_parameters",
            nonempty_json_option(&config.target_parameters),
        ),
        (
            "trainable_token_indices",
            nonempty_json_option(&config.trainable_token_indices),
        ),
        (
            "alora_invocation_tokens",
            nonempty_json_option(&config.alora_invocation_tokens),
        ),
        ("arrow_config", nonempty_json_option(&config.arrow_config)),
        ("corda_config", nonempty_json_option(&config.corda_config)),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        bail!(
            "unsupported nonstandard LoRA feature(s): {}",
            unsupported.join(", ")
        );
    }
    for (name, rank) in &config.rank_pattern {
        if name.trim().is_empty() || *rank == 0 {
            bail!("LoRA rank_pattern contains an empty name or zero rank");
        }
    }
    for (name, alpha) in &config.alpha_pattern {
        if name.trim().is_empty() || !alpha.is_finite() || *alpha <= 0.0 {
            bail!("LoRA alpha_pattern contains an empty name or invalid alpha");
        }
    }
    Ok(())
}

fn nonempty_json_option(value: &Option<serde_json::Value>) -> bool {
    match value {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(false)) => false,
        Some(serde_json::Value::Array(values)) => !values.is_empty(),
        Some(serde_json::Value::Object(values)) => !values.is_empty(),
        Some(serde_json::Value::String(value)) => !value.is_empty(),
        Some(_) => true,
    }
}

fn reject_pickle_adapter_files(adapter_dir: &Path) -> Result<()> {
    for filename in ["adapter_model.bin", "adapter_model.pt", "pytorch_model.bin"] {
        let path = adapter_dir.join(filename);
        if path.exists() {
            bail!(
                "unsafe pickle-based adapter weights are not supported: {}; re-export as {}",
                path.display(),
                HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME
            );
        }
    }
    Ok(())
}

fn validate_bound_manifest(
    base_weights: &Path,
    config_path: &Path,
    adapter_weights: &Path,
    manifest_path: &Path,
    base_sha256: &str,
) -> Result<String> {
    let checksum_path = PathBuf::from(format!("{}.sha256", manifest_path.display()));
    let expected_manifest_sha = fs::read_to_string(&checksum_path)
        .with_context(|| {
            format!(
                "reading adapter manifest checksum {}",
                checksum_path.display()
            )
        })?
        .split_whitespace()
        .next()
        .context("adapter manifest checksum file is empty")?
        .to_ascii_lowercase();
    if !is_sha256(&expected_manifest_sha) {
        bail!("adapter manifest checksum is not a valid SHA-256 digest");
    }
    let actual_manifest_sha = sha256_file(manifest_path)?;
    if actual_manifest_sha != expected_manifest_sha {
        bail!("adapter manifest SHA-256 verification failed; refusing to merge");
    }
    let manifest_bytes = fs::read(manifest_path)
        .with_context(|| format!("reading adapter manifest {}", manifest_path.display()))?;
    let manifest: LoraAdapterManifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parsing adapter manifest {}", manifest_path.display()))?;
    if manifest.manifest_version != HIERARCHOS_LORA_ADAPTER_MANIFEST_VERSION {
        bail!(
            "unsupported adapter manifest version {}",
            manifest.manifest_version
        );
    }
    if manifest.format != HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT {
        bail!("unsupported adapter manifest format {:?}", manifest.format);
    }
    let base_filename = base_weights
        .file_name()
        .and_then(|name| name.to_str())
        .context("base model checkpoint filename is not valid UTF-8")?;
    if manifest.base_checkpoint.filename != base_filename
        || !manifest
            .base_checkpoint
            .sha256
            .eq_ignore_ascii_case(base_sha256)
    {
        bail!("LoRA adapter is not bound to this exact base checkpoint");
    }
    for (filename, path) in [
        (HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME, config_path),
        (HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME, adapter_weights),
    ] {
        let expected = manifest
            .adapter_files
            .get(filename)
            .with_context(|| format!("adapter manifest has no hash for {filename}"))?;
        let actual = sha256_file(path)?;
        if !expected.eq_ignore_ascii_case(&actual) {
            bail!("LoRA adapter file hash mismatch for {filename}");
        }
    }
    if !is_sha256(&manifest.architecture_contract_sha256) {
        bail!("adapter manifest architecture contract SHA-256 is invalid");
    }
    let base_contract_hash = read_base_architecture_contract_hash(base_weights)?;
    if !manifest
        .architecture_contract_sha256
        .eq_ignore_ascii_case(&base_contract_hash)
    {
        bail!("LoRA adapter architecture contract differs from the selected base");
    }
    Ok(base_contract_hash)
}

fn read_base_architecture_contract_hash(base_weights: &Path) -> Result<String> {
    let model_dir = base_weights.parent().unwrap_or_else(|| Path::new("."));
    for filename in ["hierarchos_rust_config.json", "hierarchos_config.json"] {
        let path = model_dir.join(filename);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        if let Some(hash) = value
            .get("architecture_contract_sha256")
            .and_then(serde_json::Value::as_str)
        {
            let hash = hash.to_ascii_lowercase();
            if is_sha256(&hash) {
                return Ok(hash);
            }
            bail!(
                "{} contains an invalid architecture contract hash",
                path.display()
            );
        }
    }
    bail!(
        "base model package has no architecture_contract_sha256 in hierarchos_rust_config.json or hierarchos_config.json"
    )
}

fn discover_adapter_tensors(
    tensors: &SafeTensors<'_>,
    config: &LoraAdapterConfig,
) -> Result<(Vec<AdapterPair>, Vec<String>)> {
    let mut a_keys = BTreeMap::<String, String>::new();
    let mut b_keys = BTreeMap::<String, String>::new();
    let mut saved_module_keys = Vec::new();
    let saves_ltm = config.modules_to_save.iter().any(|module| module == "ltm");

    for key in tensors.names() {
        if !key.starts_with(PEFT_STATE_PREFIX) {
            bail!("unsupported adapter tensor key {key:?}; expected prefix {PEFT_STATE_PREFIX:?}");
        }
        let tensor = tensors.tensor(key)?;
        if tensor.data().is_empty() {
            bail!("PEFT adapter tensor {key:?} is empty");
        }
        if let Some(prefix) = key.strip_suffix(LORA_A_SUFFIX) {
            let module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("LoRA A tensor lost validated PEFT prefix")?;
            validate_target_module(module_name, config)?;
            if a_keys.insert(prefix.to_string(), key.to_string()).is_some() {
                bail!("duplicate LoRA A tensor for module {module_name:?}");
            }
        } else if let Some(prefix) = key.strip_suffix(LORA_B_SUFFIX) {
            let module_name = prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .context("LoRA B tensor lost validated PEFT prefix")?;
            validate_target_module(module_name, config)?;
            if b_keys.insert(prefix.to_string(), key.to_string()).is_some() {
                bail!("duplicate LoRA B tensor for module {module_name:?}");
            }
        } else if saves_ltm && key.starts_with(&format!("{PEFT_STATE_PREFIX}ltm.")) {
            // Decode now for dtype/non-finite validation even though replacement
            // is materialized after pair discovery.
            let _ = decode_float_tensor(key, &tensor)?;
            saved_module_keys.push(key.to_string());
        } else {
            bail!(
                "unsupported tensor {key:?} in adapter; expected standard LoRA A/B matrices{}",
                if saves_ltm {
                    " or saved ltm tensors"
                } else {
                    ""
                }
            );
        }
    }

    if a_keys.is_empty() || a_keys.keys().ne(b_keys.keys()) {
        let missing_b = a_keys
            .keys()
            .filter(|key| !b_keys.contains_key(*key))
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        let missing_a = b_keys
            .keys()
            .filter(|key| !a_keys.contains_key(*key))
            .take(8)
            .cloned()
            .collect::<Vec<_>>();
        bail!(
            "PEFT adapter has incomplete LoRA A/B pairs; missing_B={missing_b:?} missing_A={missing_a:?}"
        );
    }
    let pairs = a_keys
        .into_iter()
        .map(|(prefix, a_key)| AdapterPair {
            module_name: prefix
                .strip_prefix(PEFT_STATE_PREFIX)
                .expect("prefix validated above")
                .to_string(),
            a_key,
            b_key: b_keys
                .get(&prefix)
                .expect("A/B key sets validated above")
                .clone(),
        })
        .collect::<Vec<_>>();
    Ok((pairs, saved_module_keys))
}

fn validate_target_module(module_name: &str, config: &LoraAdapterConfig) -> Result<()> {
    if config
        .target_modules
        .iter()
        .any(|target| module_name == target || module_name.ends_with(&format!(".{target}")))
    {
        return Ok(());
    }
    bail!("LoRA tensor module {module_name:?} is not declared by adapter target_modules")
}

fn pattern_value<'a, T>(module_name: &str, pattern: &'a BTreeMap<String, T>) -> Option<&'a T> {
    pattern
        .iter()
        .filter(|(name, _)| {
            module_name == name.as_str() || module_name.ends_with(&format!(".{name}"))
        })
        .max_by_key(|(name, _)| name.len())
        .map(|(_, value)| value)
}

fn add_lora_delta(
    base: &mut [f32],
    a: &[f32],
    b: &[f32],
    out_dim: usize,
    rank: usize,
    in_dim: usize,
    scale: f32,
    transpose: bool,
) -> Result<()> {
    let a_len = rank.checked_mul(in_dim).context("LoRA A size overflow")?;
    let b_len = out_dim.checked_mul(rank).context("LoRA B size overflow")?;
    let base_len = out_dim
        .checked_mul(in_dim)
        .context("LoRA merged tensor size overflow")?;
    if a.len() != a_len || b.len() != b_len || base.len() != base_len {
        bail!(
            "LoRA matrix payload lengths do not match geometry: base={} A={} B={} expected base={base_len} A={a_len} B={b_len}",
            base.len(),
            a.len(),
            b.len()
        );
    }
    for out in 0..out_dim {
        for input in 0..in_dim {
            let mut sum = 0.0f32;
            for inner in 0..rank {
                sum += b[out * rank + inner] * a[inner * in_dim + input];
            }
            let index = if transpose {
                input * out_dim + out
            } else {
                out * in_dim + input
            };
            base[index] += sum * scale;
            if !base[index].is_finite() {
                bail!("LoRA merge produced a non-finite model weight");
            }
        }
    }
    Ok(())
}

fn decode_float_tensor(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let raw = tensor.data();
    let mut values = Vec::with_capacity(match tensor.dtype() {
        Dtype::F32 => raw.len() / 4,
        Dtype::F16 | Dtype::BF16 => raw.len() / 2,
        dtype => bail!("tensor {name:?} must be F32/F16/BF16, got {dtype:?}"),
    });
    match tensor.dtype() {
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                bail!("tensor {name:?} has invalid FP32 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid FP16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                bail!("tensor {name:?} has invalid BF16 byte length {}", raw.len());
            }
            for chunk in raw.chunks_exact(2) {
                values.push(f32::from_bits(
                    u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16,
                ));
            }
        }
        _ => unreachable!("dtype validated above"),
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("tensor {name:?} contains non-finite floating-point values");
    }
    Ok(values)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut mantissa = u32::from(fraction);
                let mut shift = 0u32;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    shift += 1;
                }
                mantissa &= 0x03ff;
                let exp = 127u32 - 15 - shift + 1;
                sign | (exp << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (u32::from(fraction) << 13),
        _ => {
            let exp = u32::from(exponent) + (127 - 15);
            sign | (exp << 23) | (u32::from(fraction) << 13)
        }
    };
    f32::from_bits(value)
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use safetensors::{serialize_to_file, tensor::TensorView};
    use serde_json::json;

    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hierarchos-native-lora-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn write_tensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Result<()> {
        let bytes = tensors
            .iter()
            .map(|(_, _, values)| f32_bytes(values))
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .zip(&bytes)
            .map(|((name, shape, _), bytes)| {
                Ok((*name, TensorView::new(Dtype::F32, shape.clone(), bytes)?))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, path)?;
        Ok(())
    }

    fn write_bound_manifest(base: &Path, adapter_dir: &Path, arch_hash: &str) -> Result<()> {
        let config = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME);
        let weights = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME);
        let manifest_path = adapter_dir.join(HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME);
        let manifest = json!({
            "manifest_version": 1,
            "format": HIERARCHOS_LORA_ADAPTER_MANIFEST_FORMAT,
            "base_checkpoint": {
                "filename": base.file_name().unwrap().to_string_lossy(),
                "sha256": sha256_file(base)?,
            },
            "architecture_contract_sha256": arch_hash,
            "adapter_files": {
                HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME: sha256_file(&config)?,
                HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME: sha256_file(&weights)?,
            },
            "finetune_run_identity": {},
            "tokenizer_identity": {},
            "lora_geometry": {},
        });
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let checksum = sha256_file(&manifest_path)?;
        fs::write(
            format!("{}.sha256", manifest_path.display()),
            format!("{checksum}  {HIERARCHOS_LORA_ADAPTER_MANIFEST_FILENAME}\n"),
        )?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_applies_b_times_a_and_saved_ltm() -> Result<()> {
        let root = temp_dir("merge");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "1".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(
            &base,
            &[
                (
                    "block.key.weight",
                    vec![2, 3],
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ),
                ("ltm.key_proj.weight", vec![1, 2], vec![7.0, 8.0]),
                ("untouched.weight", vec![1], vec![9.0]),
            ],
        )?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec_pretty(&json!({
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.05,
                "target_modules": ["key"],
                "modules_to_save": ["ltm"],
                "bias": "none",
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                (
                    "base_model.model.block.key.lora_A.weight",
                    vec![2, 3],
                    vec![1.0, 0.0, 2.0, 0.0, 1.0, 1.0],
                ),
                (
                    "base_model.model.block.key.lora_B.weight",
                    vec![2, 2],
                    vec![1.0, 2.0, 3.0, 4.0],
                ),
                (
                    "base_model.model.ltm.key_proj.weight",
                    vec![1, 2],
                    vec![10.0, 11.0],
                ),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;

        let report = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)?;
        assert_eq!(report.merged_lora_modules, 1);
        assert_eq!(report.replaced_module_tensors, 1);
        let (_, merged) = crate::checkpoint::read_f32_tensor(&output, "block.key.weight")?;
        // B@A = [[1,2,4],[3,4,10]], scale=alpha/r=2.
        assert_eq!(merged, vec![3.0, 6.0, 11.0, 10.0, 13.0, 26.0]);
        let (_, ltm) = crate::checkpoint::read_f32_tensor(&output, "ltm.key_proj.weight")?;
        assert_eq!(ltm, vec![10.0, 11.0]);
        let (_, untouched) = crate::checkpoint::read_f32_tensor(&output, "untouched.weight")?;
        assert_eq!(untouched, vec![9.0]);
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn native_lora_merge_rejects_unbound_base() -> Result<()> {
        let root = temp_dir("binding");
        let base_dir = root.join("base");
        let adapter_dir = root.join("adapter");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&adapter_dir)?;
        let base = base_dir.join("model.safetensors");
        let output = root.join("merged.safetensors");
        let arch_hash = "2".repeat(64);
        fs::write(
            base_dir.join("hierarchos_rust_config.json"),
            serde_json::to_vec(&json!({"architecture_contract_sha256": arch_hash}))?,
        )?;
        write_tensors(&base, &[("key.weight", vec![1, 1], vec![1.0])])?;
        fs::write(
            adapter_dir.join(HIERARCHOS_LORA_ADAPTER_CONFIG_FILENAME),
            serde_json::to_vec(&json!({
                "peft_type":"LORA", "task_type":"CAUSAL_LM", "r":1,
                "lora_alpha":1, "target_modules":["key"], "bias":"none"
            }))?,
        )?;
        write_tensors(
            &adapter_dir.join(HIERARCHOS_LORA_ADAPTER_WEIGHTS_FILENAME),
            &[
                ("base_model.model.key.lora_A.weight", vec![1, 1], vec![1.0]),
                ("base_model.model.key.lora_B.weight", vec![1, 1], vec![1.0]),
            ],
        )?;
        write_bound_manifest(&base, &adapter_dir, &arch_hash)?;
        // Mutate the base after the adapter has been cryptographically bound.
        write_tensors(&base, &[("key.weight", vec![1, 1], vec![2.0])])?;
        let error = merge_hierarchos_lora_safetensors(&base, &adapter_dir, &output)
            .expect_err("base binding mismatch must fail");
        assert!(error.to_string().contains("not bound to this exact base"));
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
