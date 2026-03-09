# Session: phase2-architecture

## Solution Space
**Updated:** 2026-03-09

**Problem:** Phase 2 (Model Loading & Architecture) is complete, but LoRA is applied at the wrong granularity (layer output level instead of per-projection). This will undermine Phase 3 (LoRA Training) quality and performance.

**Key Constraint:** Phase 3 depends on correct per-projection LoRA integration (Q, K, V, O, Gate, Up, Down).

### Selected Approach: LoRA-Aware Projection Layer (Reframe)

Replace raw `Linear` in Attention/MLP with a `Projection` type that optionally wraps a LoRA delta. Forward pass becomes `proj(x) + lora_delta(x)` at each site.

**Why this approach:**
- Eliminates `Option<&LoRAAdapter>` parameter threading through every `forward()` call
- Applies LoRA at the correct granularity (per-projection)
- Naturally composes with existing fused Metal LoRA kernel
- Model code stays clean — Attention/MLP don't know about LoRA

**Accepted trade-offs:**
- Requires modifying Attention and MLP constructors
- One level of indirection in Projection wrapper (negligible vs matmul cost)
- Existing tests need minor updates

### Additional Issues to Address
1. `inspect()` reads entire file into memory — should read only safetensors header
2. `num_parameters()` uses hardcoded 1M per layer — should compute from actual weights
3. No memory-mapped loading for large models

### Implementation Sketch

```rust
struct Projection {
    linear: Linear,
    lora: Option<LoRALayer>,
}

impl Projection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.linear.forward(x)?;
        if let Some(lora) = &self.lora {
            Ok((out + lora.forward(x)?)?)
        } else {
            Ok(out)
        }
    }
}
```

### Success Criteria
- [ ] `Projection` type wraps `Linear` + optional `LoRA`
- [ ] `Attention` uses `Projection` for Q, K, V, O
- [ ] `MLP` uses `Projection` for Gate, Up, Down
- [ ] `QwenDecoderLayer::forward` no longer takes `lora_adapter` parameter
- [ ] `ApplyAdapter` sets LoRA at projection level, not layer level
- [ ] Fused Metal LoRA kernel can be dispatched from `Projection::forward`
- [ ] All existing tests pass (with updates)
- [ ] `inspect()` reads only safetensors header, not full file
- [ ] `num_parameters()` computes from actual weight shapes
