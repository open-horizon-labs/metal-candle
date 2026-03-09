//! Custom operations using Candle's `CustomOp` framework.
//!
//! This module implements high-performance fused operations using Candle's
//! `CustomOp` traits, providing clean Metal buffer access and proper integration
//! with Candle's autodiff system.

// Allow similar_names for LoRA operations - A/B matrix naming is standard ML convention
#![allow(clippy::similar_names)]

use crate::backend::metal_kernels::{
    LayerNormParams, LoRAParams, MetalKernelCompiler, RMSNormParams, SoftmaxParams,
};
use candle_core::backend::BackendStorage;
use candle_core::{CustomOp1, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use std::sync::Mutex;

/// Fused `LoRA` forward pass operation.
///
/// Implements `LoRA` forward pass as a single Metal kernel:
/// `output = (input @ lora_a @ lora_b) * scaling`
///
/// This fuses two matrix multiplications and a scaling operation,
/// avoiding intermediate allocations and reducing kernel launch overhead.
pub struct FusedLoRAOp {
    /// First `LoRA` matrix (`in_features` × rank)
    lora_a: Tensor,
    /// Second `LoRA` matrix (rank × `out_features`)
    lora_b: Tensor,
    /// Scaling factor (alpha / rank)
    scaling: f32,
    /// Cached Metal compute pipeline (compiled kernel)
    pipeline: Mutex<Option<ComputePipeline>>,
}

impl FusedLoRAOp {
    /// Creates a new fused `LoRA` operation.
    ///
    /// # Errors
    ///
    /// Returns error if tensor shapes are incompatible.
    pub fn new(lora_a: Tensor, lora_b: Tensor, scaling: f32) -> Result<Self> {
        let a_dims = lora_a.dims();
        let b_dims = lora_b.dims();

        if a_dims.len() != 2 || b_dims.len() != 2 {
            candle_core::bail!(
                "LoRA matrices must be 2D, got shapes {:?} and {:?}",
                a_dims,
                b_dims
            );
        }

        if a_dims[1] != b_dims[0] {
            candle_core::bail!(
                "Incompatible LoRA matrix dimensions: {}×{} and {}×{}",
                a_dims[0],
                a_dims[1],
                b_dims[0],
                b_dims[1]
            );
        }

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            pipeline: Mutex::new(None),
        })
    }

    fn compute_output_shape(&self, input_shape: &Shape) -> Result<Shape> {
        let dims = input_shape.dims();
        if dims.is_empty() {
            candle_core::bail!("Input must have at least 1 dimension");
        }

        let in_features = dims[dims.len() - 1];
        let expected_in = self.lora_a.dim(0)?;

        if in_features != expected_in {
            candle_core::bail!(
                "Input feature dimension mismatch: expected {}, got {}",
                expected_in,
                in_features
            );
        }

        let mut output_dims = dims.to_vec();
        output_dims[dims.len() - 1] = self.lora_b.dim(1)?;

        Ok(Shape::from(output_dims))
    }

    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipeline> {
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        let compiler = MetalKernelCompiler::new(device.device()).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create compiler: {e}"))
        })?;

        let pipeline = compiler
            .create_pipeline("fused_lora_forward_tiled")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?;

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
        *guard = Some(pipeline.clone());

        Ok(pipeline)
    }

    fn metal_fwd_impl(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let output_shape = self.compute_output_shape(layout.shape())?;

        let input_dims = layout.shape().dims();
        let batch_size = if input_dims.len() == 3 {
            input_dims[0]
        } else {
            1
        };
        let seq_len = if input_dims.len() >= 2 {
            input_dims[input_dims.len() - 2]
        } else {
            1
        };
        let in_features = input_dims[input_dims.len() - 1];
        let rank = self.lora_a.dim(1)?;
        let out_features = self.lora_b.dim(1)?;

        #[allow(clippy::cast_possible_truncation)]
        let params = LoRAParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            in_features: in_features as u32,
            rank: rank as u32,
            out_features: out_features as u32,
            scaling: self.scaling,
        };

        // Get Metal buffers
        let input_buffer = storage.buffer();

        let weight_a_guard = self.lora_a.storage_and_layout();
        let candle_core::Storage::Metal(weight_a_storage) = &*weight_a_guard.0 else {
            candle_core::bail!("LoRA_A must be on Metal device")
        };
        let lora_a_buffer = weight_a_storage.buffer();

        let lora_b_guard = self.lora_b.storage_and_layout();
        let candle_core::Storage::Metal(lora_b_storage) = &*lora_b_guard.0 else {
            candle_core::bail!("LoRA_B must be on Metal device")
        };
        let lora_b_buffer = lora_b_storage.buffer();

        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_lora_output")?;

        let pipeline = self.get_or_compile_pipeline(device)?;

        {
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);

            candle_metal_kernels::utils::set_param(&encoder, 0, input_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 1, lora_a_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 2, lora_b_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 3, output_buffer.as_ref());
            encoder.set_bytes(4, &params);

            let grid_size = objc2_metal::MTLSize {
                width: batch_size,
                height: seq_len,
                depth: out_features,
            };

            let threadgroup_size = objc2_metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
            // encoder drops here, ending encoding automatically
        }

        let output_storage = MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

impl CustomOp1 for FusedLoRAOp {
    fn name(&self) -> &'static str {
        "fused_lora_forward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!(
            "FusedLoRAOp is Metal-only. Use standard LoRA operations on CPU."
        )
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        self.metal_fwd_impl(storage, layout)
    }
}

/// Fused RMS Normalization operation.
///
/// Implements RMS norm as a single Metal kernel with threadgroup reductions:
/// `rms_norm(x) = x / sqrt(mean(x^2) + eps)`
pub struct FusedRMSNormOp {
    eps: f32,
    pipeline: Mutex<Option<ComputePipeline>>,
}

impl FusedRMSNormOp {
    /// Creates a new fused RMS norm operation.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for API consistency.
    pub fn new(eps: f32) -> Result<Self> {
        Ok(Self {
            eps,
            pipeline: Mutex::new(None),
        })
    }

    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipeline> {
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        let compiler = MetalKernelCompiler::new(device.device()).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create compiler: {e}"))
        })?;

        let pipeline = compiler
            .create_pipeline("fused_rms_norm")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?;

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
        *guard = Some(pipeline.clone());

        Ok(pipeline)
    }
}

impl CustomOp1 for FusedRMSNormOp {
    fn name(&self) -> &'static str {
        "fused-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Fused RMS Norm kernel not implemented for CPU. Use Metal device.")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let pipeline = self.get_or_compile_pipeline(device)?;

        let input_dims = layout.shape().dims();
        if input_dims.is_empty() {
            candle_core::bail!(
                "Input tensor for FusedRMSNormOp must have at least 1 dimension, got {:?}",
                input_dims
            );
        }

        let (batch_size, seq_len, dim) = match input_dims.len() {
            1 => (1, 1, input_dims[0]),
            2 => (1, input_dims[0], input_dims[1]),
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            _ => candle_core::bail!(
                "RMS Norm only supports 1D, 2D, or 3D tensors, got {:?}",
                input_dims
            ),
        };

        let input_buffer = storage.buffer();
        let output_shape = layout.shape().clone();
        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_rms_norm_output")?;

        #[allow(clippy::cast_possible_truncation)]
        let params = RMSNormParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            dim: dim as u32,
            eps: self.eps,
        };

        {
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);

            candle_metal_kernels::utils::set_param(&encoder, 0, input_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 1, output_buffer.as_ref());
            encoder.set_bytes(2, &params);
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            let grid_size = objc2_metal::MTLSize {
                width: 1,
                height: seq_len,
                depth: batch_size,
            };
            let threadgroup_size = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        let output_storage = MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

/// Fused Softmax operation.
///
/// Implements softmax as a single Metal kernel with threadgroup reductions:
/// `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`
pub struct FusedSoftmaxOp {
    pipeline: Mutex<Option<ComputePipeline>>,
}

impl FusedSoftmaxOp {
    /// Creates a new fused softmax operation.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for API consistency.
    pub fn new() -> Result<Self> {
        Ok(Self {
            pipeline: Mutex::new(None),
        })
    }

    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipeline> {
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        let compiler = MetalKernelCompiler::new(device.device()).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create compiler: {e}"))
        })?;

        let pipeline = compiler
            .create_pipeline("fused_softmax")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {e}")))?;

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
        *guard = Some(pipeline.clone());

        Ok(pipeline)
    }
}

impl CustomOp1 for FusedSoftmaxOp {
    fn name(&self) -> &'static str {
        "fused-softmax"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("Fused Softmax kernel not implemented for CPU. Use Metal device.")
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let device = storage.device();
        let pipeline = self.get_or_compile_pipeline(device)?;

        let input_dims = layout.shape().dims();
        if input_dims.is_empty() {
            candle_core::bail!(
                "Input tensor for FusedSoftmaxOp must have at least 1 dimension, got {:?}",
                input_dims
            );
        }

        let (batch_size, seq_len, dim) = match input_dims.len() {
            1 => (1, 1, input_dims[0]),
            2 => (1, input_dims[0], input_dims[1]),
            3 => (input_dims[0], input_dims[1], input_dims[2]),
            _ => candle_core::bail!(
                "Softmax only supports 1D, 2D, or 3D tensors, got {:?}",
                input_dims
            ),
        };

        let input_buffer = storage.buffer();
        let output_shape = layout.shape().clone();
        let output_elem_count = output_shape.elem_count();
        let output_buffer =
            device.new_buffer(output_elem_count, storage.dtype(), "fused_softmax_output")?;

        #[allow(clippy::cast_possible_truncation)]
        let params = SoftmaxParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            dim: dim as u32,
        };

        {
            let encoder = device.command_encoder()?;
            encoder.set_compute_pipeline_state(&pipeline);

            candle_metal_kernels::utils::set_param(&encoder, 0, input_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 1, output_buffer.as_ref());
            encoder.set_bytes(2, &params);
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            let grid_size = objc2_metal::MTLSize {
                width: 1,
                height: seq_len,
                depth: batch_size,
            };
            let threadgroup_size = objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        let output_storage = MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            storage.dtype(),
        );

        Ok((output_storage, output_shape))
    }
}

/// Fused Layer Normalization operation.
///
/// Implements standard layer normalization as a single Metal kernel:
/// `normalized[i] = (input[i] - mean) / sqrt(variance + eps)`
pub struct LayerNormOp {
    eps: f64,
    pipeline: Mutex<Option<ComputePipeline>>,
}

impl LayerNormOp {
    /// Creates a new layer normalization operation.
    #[must_use]
    pub fn new(eps: f64) -> Self {
        Self {
            eps,
            pipeline: Mutex::new(None),
        }
    }

    fn get_or_compile_pipeline(
        &self,
        device: &candle_core::MetalDevice,
    ) -> Result<ComputePipeline> {
        if let Some(pipeline) = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?
            .as_ref()
        {
            return Ok(pipeline.clone());
        }

        let compiler = MetalKernelCompiler::new(device.device()).map_err(|e| {
            candle_core::Error::Msg(format!("Metal compiler init failed: {e}"))
        })?;

        let pipeline = compiler
            .create_pipeline("layer_norm")
            .map_err(|e| candle_core::Error::Msg(format!("Pipeline creation failed: {e}")))?;

        let mut guard = self
            .pipeline
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline: {e}")))?;
        *guard = Some(pipeline.clone());

        Ok(pipeline)
    }
}

impl CustomOp1 for LayerNormOp {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        let shape = layout.shape();
        if shape.rank() != 2 {
            candle_core::bail!("LayerNorm expects 2D tensors, got shape {:?}", shape.dims());
        }

        let (batch_size, hidden_size) = (shape.dims()[0], shape.dims()[1]);
        let input = storage.as_slice::<f32>()?;

        let mut output = vec![0.0f32; batch_size * hidden_size];

        for b in 0..batch_size {
            let offset = b * hidden_size;
            let row = &input[offset..offset + hidden_size];

            #[allow(clippy::cast_precision_loss)]
            let hidden_size_f32 = hidden_size as f32;
            let mean: f32 = row.iter().sum::<f32>() / hidden_size_f32;

            let variance: f32 =
                row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / hidden_size_f32;

            #[allow(clippy::cast_possible_truncation)]
            let inv_std = 1.0 / (variance + self.eps as f32).sqrt();

            for (i, &val) in row.iter().enumerate() {
                output[offset + i] = (val - mean) * inv_std;
            }
        }

        let storage = candle_core::CpuStorage::F32(output);
        Ok((storage, shape.clone()))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        let shape = layout.shape();
        if shape.rank() != 2 {
            candle_core::bail!("LayerNorm expects 2D tensors, got shape {:?}", shape.dims());
        }

        let (batch_size, hidden_size) = (shape.dims()[0], shape.dims()[1]);
        let device = storage.device();

        let pipeline = self.get_or_compile_pipeline(device)?;

        let output_elem_count = batch_size * hidden_size;
        let output_buffer =
            device.new_buffer(output_elem_count, candle_core::DType::F32, "layer_norm_output")?;

        #[allow(clippy::cast_possible_truncation)]
        let params = LayerNormParams {
            batch_size: batch_size as u32,
            hidden_size: hidden_size as u32,
            eps: self.eps as f32,
        };

        {
            let input_buffer = storage.buffer();
            let encoder = device.command_encoder()?;

            encoder.set_compute_pipeline_state(&pipeline);
            candle_metal_kernels::utils::set_param(&encoder, 0, input_buffer);
            candle_metal_kernels::utils::set_param(&encoder, 1, output_buffer.as_ref());
            encoder.set_bytes(2, &params);
            encoder.set_threadgroup_memory_length(0, 256 * 4);

            let grid_size = objc2_metal::MTLSize {
                width: batch_size,
                height: 1,
                depth: 1,
            };
            let threadgroup_size = objc2_metal::MTLSize {
                width: 256.min(hidden_size),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        }

        let output_storage = MetalStorage::new(
            output_buffer,
            device.clone(),
            output_elem_count,
            candle_core::DType::F32,
        );

        Ok((output_storage, shape.clone()))
    }
}

/// Convenience function to apply layer normalization to a tensor.
///
/// # Errors
///
/// Returns error if the operation fails or if the tensor is not 2D.
pub fn layer_norm(tensor: &Tensor, eps: f64) -> Result<Tensor> {
    tensor.apply_op1(LayerNormOp::new(eps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device as MetalCandleDevice;
    use candle_core::DType;

    #[test]
    fn test_fused_lora_op_creation() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| MetalCandleDevice::new_metal(0)) else {
            return;
        };

        let candle_device = device.as_candle_device();
        let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), candle_device).unwrap();
        let lora_b = Tensor::zeros((8, 512), DType::F32, candle_device).unwrap();

        let op = FusedLoRAOp::new(lora_a, lora_b, 2.0);
        assert!(op.is_ok());
    }

    #[test]
    fn test_fused_lora_op_invalid_dimensions() {
        let Ok(Ok(device)) = std::panic::catch_unwind(|| MetalCandleDevice::new_metal(0)) else {
            return;
        };

        let candle_device = device.as_candle_device();
        let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), candle_device).unwrap();
        let lora_b = Tensor::zeros((16, 512), DType::F32, candle_device).unwrap();

        let op = FusedLoRAOp::new(lora_a, lora_b, 2.0);
        assert!(op.is_err());
    }
}
