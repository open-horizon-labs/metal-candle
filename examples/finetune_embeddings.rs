//! Fine-tune embedding models with LoRA for domain-specific retrieval.
//!
//! Downloads E5-small-v2, applies LoRA to attention Q/K/V, and trains
//! on query-document pairs using contrastive loss.
//!
//! Run with: `cargo run --release --features embeddings --example finetune_embeddings`

use anyhow::Result;
use candle_core::Device;
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
use metal_candle::training::{contrastive_loss, AdamW, AdamWConfig, LoRAConfig, LRScheduler};

fn main() -> Result<()> {
    println!("=== Embedding Model LoRA Fine-Tuning ===\n");

    // CPU for training (Metal autograd limitation)
    let device = Device::Cpu;

    // 1. Load embedding model
    println!("Loading E5-small-v2...");
    let mut model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device.clone())?;
    println!("Dimension: {}\n", model.dimension());

    // 2. Apply LoRA to attention Q/K/V
    let lora_config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };
    model.apply_lora(&lora_config)?;

    let trainable_vars: Vec<candle_core::Var> = model.lora_vars().into_iter().cloned().collect();
    println!("LoRA trainable vars: {}", trainable_vars.len());
    let num_params: usize = trainable_vars.iter().map(|v| v.as_tensor().elem_count()).sum();
    println!("LoRA parameters: {num_params}\n");

    // 3. Training data: (query, positive_document) pairs
    // In production, load from your domain corpus
    let pairs = [
        ("How to sort a vector in Rust?", "Use vec.sort() or vec.sort_by() for custom ordering"),
        ("Rust error handling", "Use Result<T, E> with the ? operator for propagating errors"),
        ("What is ownership in Rust?", "Ownership is Rust's system for managing memory without a garbage collector"),
        ("How to create a HashMap?", "Use std::collections::HashMap::new() or the maplit crate for literal syntax"),
        ("Async programming in Rust", "Use async/await with tokio or async-std runtime for concurrent operations"),
        ("Pattern matching examples", "Use match expressions with enum variants, guards, and destructuring"),
        ("Trait implementations", "impl TraitName for StructName to provide behavior for your types"),
        ("Lifetime annotations", "Use 'a syntax to tell the compiler how references relate to each other"),
    ];

    // 4. Training loop
    let num_epochs = 5;
    let total_steps = num_epochs * 1; // 1 batch per epoch (all pairs at once)
    let scheduler = LRScheduler::WarmupCosine {
        warmup_steps: 2,
        max_lr: 1e-4,
        min_lr: 1e-6,
        total_steps,
    };
    let mut optimizer = AdamW::new(AdamWConfig::default())?;
    let scale = 20.0; // Temperature for contrastive loss

    println!("Training: {num_epochs} epochs, {} pairs\n", pairs.len());

    let queries: Vec<&str> = pairs.iter().map(|(q, _)| *q).collect();
    let documents: Vec<&str> = pairs.iter().map(|(_, d)| *d).collect();

    for epoch in 0..num_epochs {
        let start = std::time::Instant::now();

        // Encode queries and documents
        let query_embs = model.encode(&queries)?;
        let doc_embs = model.encode(&documents)?;

        // Contrastive loss
        let loss = contrastive_loss(&query_embs, &doc_embs, scale)?;
        let loss_val = loss.to_vec0::<f32>()?;

        // Backward
        let grads = loss.backward()?;

        // Update LoRA parameters
        let lr = scheduler.get_lr(epoch);
        optimizer.set_lr(lr);
        let mut updated = 0;
        for var in &trainable_vars {
            if let Some(grad) = grads.get(var) {
                optimizer.step_var(var, &grad)?;
                updated += 1;
            }
        }

        println!(
            "Epoch {}/{}: loss={:.4}, lr={:.2e}, grads={}/{}, {:.2}s",
            epoch + 1,
            num_epochs,
            loss_val,
            lr,
            updated,
            trainable_vars.len(),
            start.elapsed().as_secs_f32(),
        );
    }

    // 5. Save LoRA weights
    let lora_path = std::path::Path::new("e5_lora_finetune.safetensors");
    {
        let tensors: std::collections::HashMap<String, candle_core::Tensor> = trainable_vars
            .iter()
            .enumerate()
            .map(|(i, var)| (format!("lora.{i}"), var.as_tensor().clone()))
            .collect();
        candle_core::safetensors::save(&tensors, lora_path)?;
        println!("\nLoRA checkpoint saved to {}", lora_path.display());
    }

    // 6. Verify: load into a fresh model and check similarity
    println!("\nLoading checkpoint into fresh model...");
    let mut fresh_model =
        EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device.clone())?;
    fresh_model.apply_lora(&lora_config)?;

    // Load saved weights
    let saved_tensors: std::collections::HashMap<String, candle_core::Tensor> =
        candle_core::safetensors::load(lora_path, &device)?;
    let fresh_vars: Vec<candle_core::Var> = fresh_model.lora_vars().into_iter().cloned().collect();
    for (i, var) in fresh_vars.iter().enumerate() {
        if let Some(t) = saved_tensors.get(&format!("lora.{i}")) {
            var.set(t)?;
        }
    }

    let q_emb = fresh_model.encode(&["How to sort in Rust?"])?;
    let d_emb = fresh_model.encode(&["Use vec.sort() for sorting"])?;
    let sim_loaded = q_emb.matmul(&d_emb.t()?)?.to_vec2::<f32>()?;
    println!("Loaded model similarity: {:.4}", sim_loaded[0][0]);

    // 7. Compare: trained vs base model
    println!("\nSimilarity check (query vs matching doc):");
    let q_emb = model.encode(&["How to sort in Rust?"])?;
    let d_emb = model.encode(&["Use vec.sort() for sorting"])?;
    let sim = q_emb.matmul(&d_emb.t()?)?.to_vec2::<f32>()?;
    println!("  'How to sort in Rust?' <-> 'Use vec.sort()...' = {:.4}", sim[0][0]);

    // Remove LoRA and check without it
    model.remove_lora();
    let q_emb = model.encode(&["How to sort in Rust?"])?;
    let d_emb = model.encode(&["Use vec.sort() for sorting"])?;
    let sim_base = q_emb.matmul(&d_emb.t()?)?.to_vec2::<f32>()?;
    println!("  Without LoRA: {:.4}", sim_base[0][0]);

    println!("\nDone!");
    Ok(())
}
