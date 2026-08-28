use std::{
    path::Path,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::mixed_precision::{VulkanParameterStorageMirror, VulkanParameterStorageMirrorRefresher};
use crate::{
    read_f32_tensor, vulkan, AdamWHyperParams, GpuBuffer, VulkanDevice,
    VulkanParameterStorageFormat,
};

const GRADIENT_ACCUMULATE_SPV: &[u8] = include_bytes!("../shaders/gradient_accumulate.spv");
const ADAMW_SPV: &[u8] = include_bytes!("../shaders/adamw.spv");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LenPush {
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdamWPush {
    len: u32,
    step: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

struct SharedLmHeadInner {
    device: VulkanDevice,
    context_dim: usize,
    vocab_size: usize,
    weight: GpuBuffer,
    accumulated_grad: GpuBuffer,
    exp_avg: GpuBuffer,
    exp_avg_sq: GpuBuffer,
    readback: GpuBuffer,
    execution_mirror: Mutex<Option<SharedLmHeadExecutionMirror>>,
    step: AtomicU32,
    gradient_accumulate: vulkan::ComputeKernel,
    adamw: vulkan::ComputeKernel,
}

struct SharedLmHeadExecutionMirror {
    mirror: VulkanParameterStorageMirror,
    refresher: VulkanParameterStorageMirrorRefresher,
}

/// One physical and optimizer identity for Hierarchos' tied
/// `lm_head.weight` parameter.
///
/// Clones share the same Vulkan weight allocation, gradient accumulator,
/// AdamW first/second moments, and step counter. This lets H-DeepEmbed,
/// L-DeepEmbed, and the LM-head loss bind the exact same parameter without
/// duplicating Vulkan handles or silently training divergent copies.
#[derive(Clone)]
pub struct SharedLmHeadParameter {
    inner: Arc<SharedLmHeadInner>,
}

impl SharedLmHeadParameter {
    pub fn new(
        device: VulkanDevice,
        context_dim: usize,
        vocab_size: usize,
        weight: &[f32],
    ) -> Result<Self> {
        if context_dim == 0 || vocab_size == 0 {
            bail!("shared lm_head context_dim and vocab_size must be positive");
        }
        let len = context_dim
            .checked_mul(vocab_size)
            .context("shared lm_head element count overflow")?;
        if weight.len() != len {
            bail!(
                "shared lm_head weight has {} values; expected {len} for [{vocab_size}, {context_dim}]",
                weight.len()
            );
        }
        if weight.iter().any(|value| !value.is_finite()) {
            bail!("shared lm_head weight contains non-finite values");
        }

        Ok(Self {
            inner: Arc::new(SharedLmHeadInner {
                gradient_accumulate: vulkan::ComputeKernel::new(
                    &device,
                    GRADIENT_ACCUMULATE_SPV,
                    2,
                    std::mem::size_of::<LenPush>() as u32,
                )?,
                adamw: vulkan::ComputeKernel::new(
                    &device,
                    ADAMW_SPV,
                    4,
                    std::mem::size_of::<AdamWPush>() as u32,
                )?,
                weight: GpuBuffer::from_f32(&device, weight)?,
                accumulated_grad: GpuBuffer::zeros_f32(&device, len)?,
                exp_avg: GpuBuffer::zeros_f32(&device, len)?,
                exp_avg_sq: GpuBuffer::zeros_f32(&device, len)?,
                readback: GpuBuffer::zeros_host_f32(&device, len)?,
                execution_mirror: Mutex::new(None),
                step: AtomicU32::new(0),
                device,
                context_dim,
                vocab_size,
            }),
        })
    }

    pub fn from_model_package(device: VulkanDevice, model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("model.safetensors");
        let (shape, weight) = read_f32_tensor(&path, "lm_head.weight")?;
        let (vocab_size, context_dim) = match shape.as_slice() {
            [vocab_size, context_dim] if *vocab_size > 0 && *context_dim > 0 => {
                (*vocab_size, *context_dim)
            }
            _ => bail!("lm_head.weight must have shape [vocab_size, context_dim], got {shape:?}"),
        };
        Self::new(device, context_dim, vocab_size, &weight)
    }

    pub fn context_dim(&self) -> usize {
        self.inner.context_dim
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    pub fn step(&self) -> u32 {
        self.inner.step.load(Ordering::Relaxed)
    }

    pub fn device_name(&self) -> &str {
        self.inner.device.name()
    }

    pub fn shares_identity_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        self.inner.weight.read_f32(self.len())
    }

    pub(crate) fn weight_buffer(&self) -> &GpuBuffer {
        &self.inner.weight
    }

    pub(crate) fn install_fp16_parameter_storage_mirror(
        &self,
        mirror: VulkanParameterStorageMirror,
    ) -> Result<()> {
        if mirror.format() != VulkanParameterStorageFormat::Fp16 {
            bail!(
                "shared lm_head execution mirror must be fp16; got {}",
                mirror.format().label()
            );
        }
        if mirror.len() != self.len() {
            bail!(
                "shared lm_head execution mirror has {} elements; expected {}",
                mirror.len(),
                self.len()
            );
        }
        let mut execution_mirror = self
            .inner
            .execution_mirror
            .lock()
            .map_err(|_| anyhow::anyhow!("shared lm_head execution mirror lock poisoned"))?;
        if let Some(existing) = execution_mirror.as_ref() {
            if !existing
                .mirror
                .packed_storage()
                .shares_allocation_with(mirror.packed_storage())
            {
                bail!("shared lm_head already has a different packed execution mirror");
            }
            return Ok(());
        }
        *execution_mirror = Some(SharedLmHeadExecutionMirror {
            refresher: VulkanParameterStorageMirrorRefresher::new(
                &self.inner.device,
                VulkanParameterStorageFormat::Fp16,
            )?,
            mirror,
        });
        Ok(())
    }

    pub(crate) fn fp16_parameter_storage_mirror(
        &self,
    ) -> Result<Option<VulkanParameterStorageMirror>> {
        let execution_mirror = self
            .inner
            .execution_mirror
            .lock()
            .map_err(|_| anyhow::anyhow!("shared lm_head execution mirror lock poisoned"))?;
        Ok(execution_mirror
            .as_ref()
            .map(|execution| execution.mirror.clone()))
    }

    pub fn fp16_parameter_storage_active(&self) -> bool {
        self.inner
            .execution_mirror
            .lock()
            .map(|mirror| mirror.is_some())
            .unwrap_or(false)
    }

    /// Diagnostic expansion of the compact execution mirror. Checkpoints and
    /// optimizer state still use `weights()` from the canonical FP32 master.
    pub fn fp16_parameter_storage_values(&self) -> Result<Option<Vec<f32>>> {
        let mirror = self.fp16_parameter_storage_mirror()?;
        mirror
            .map(|mirror| mirror.read_expanded_f32(&self.inner.device))
            .transpose()
    }

    pub(crate) fn device(&self) -> VulkanDevice {
        self.inner.device.clone()
    }

    pub(crate) fn gradient_buffer(&self) -> &GpuBuffer {
        &self.inner.accumulated_grad
    }

    pub(crate) fn record_zero_grad(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        commands.fill_zero_f32(&self.inner.accumulated_grad, self.len())
    }

    pub(crate) fn record_accumulate_gradient(
        &self,
        commands: &mut vulkan::ComputeBatch,
        gradient: &GpuBuffer,
    ) -> Result<()> {
        let push = LenPush {
            len: self.len() as u32,
        };
        self.inner.gradient_accumulate.record_dispatch(
            commands,
            &[&self.inner.accumulated_grad, gradient],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.len(), 256), 1, 1],
        )
    }

    pub(crate) fn record_step(
        &self,
        commands: &mut vulkan::ComputeBatch,
        hyper: AdamWHyperParams,
    ) -> Result<u32> {
        hyper.validate()?;
        let previous = self
            .inner
            .step
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |step| {
                step.checked_add(1)
            })
            .map_err(|_| anyhow::anyhow!("shared lm_head AdamW step overflow"))?;
        let next_step = previous + 1;
        let push = AdamWPush {
            len: self.len() as u32,
            step: next_step,
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay: hyper.weight_decay,
        };
        self.inner.adamw.record_dispatch(
            commands,
            &[
                &self.inner.weight,
                &self.inner.accumulated_grad,
                &self.inner.exp_avg,
                &self.inner.exp_avg_sq,
            ],
            bytemuck::bytes_of(&push),
            [div_ceil_u32(self.len(), 256), 1, 1],
        )?;
        let execution_mirror = self
            .inner
            .execution_mirror
            .lock()
            .map_err(|_| anyhow::anyhow!("shared lm_head execution mirror lock poisoned"))?;
        if let Some(execution) = execution_mirror.as_ref() {
            execution
                .refresher
                .record_refresh(commands, &self.inner.weight, &execution.mirror)?;
        }
        Ok(next_step)
    }

    pub(crate) fn record_readback(&self, commands: &mut vulkan::ComputeBatch) -> Result<()> {
        commands.readback_f32(&self.inner.weight, &self.inner.readback, self.len())
    }

    pub(crate) fn read_recorded_weights(&self) -> Result<Vec<f32>> {
        self.inner.readback.read_f32(self.len())
    }

    fn len(&self) -> usize {
        self.inner.context_dim * self.inner.vocab_size
    }
}

fn div_ceil_u32(value: usize, divisor: usize) -> u32 {
    value.div_ceil(divisor) as u32
}
