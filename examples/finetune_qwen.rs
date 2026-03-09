//! End-to-end LoRA fine-tuning of Qwen2.5-Coder on Apple Silicon.
//!
//! Downloads the model from HuggingFace, tokenizes training data,
//! and trains a LoRA adapter with the full training loop.
//!
//! Run with: `cargo run --release --example finetune_qwen`

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo};
use metal_candle::models::{ModelConfig, Qwen};
use metal_candle::training::{
    checkpoint::{save_checkpoint, CheckpointMetadata},
    AdamW, AdamWConfig, ApplyAdapter, LRScheduler, LoRAAdapter, LoRAAdapterConfig, TargetModule,
};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Download Qwen2.5-Coder-0.5B from HuggingFace Hub.
fn download_qwen() -> Result<std::path::PathBuf> {
    let model_id = "Qwen/Qwen2.5-Coder-0.5B";
    println!("Downloading {model_id}...");

    let api = Api::new().context("Failed to init HuggingFace API")?;
    let repo = api.repo(Repo::model(model_id.to_string()));

    let config_path = repo.get("config.json").context("config.json")?;
    repo.get("tokenizer.json").context("tokenizer.json")?;
    repo.get("model.safetensors").context("model.safetensors")?;

    Ok(config_path.parent().unwrap().to_path_buf())
}

/// Load model config, weights, and tokenizer.
fn load_model(
    model_dir: &std::path::Path,
    device: &Device,
) -> Result<(Qwen, ModelConfig, Tokenizer)> {
    let config = ModelConfig::from_file(model_dir.join("config.json"))?;
    config.validate()?;
    println!(
        "Model: {} layers, hidden={}, vocab={}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size
    );

    let tensors =
        candle_core::safetensors::load(model_dir.join("model.safetensors"), device)?;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let model = Qwen::new(&config, vb)?;
    println!("Parameters: {}", model.num_parameters());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    Ok((model, config, tokenizer))
}

/// Tokenize texts into (input_ids, target_ids) pairs for causal LM training.
fn prepare_dataset(
    texts: &[&str],
    tokenizer: &Tokenizer,
    max_len: usize,
    device: &Device,
) -> Result<Vec<(Tensor, Tensor)>> {
    let mut dataset = Vec::new();
    for text in texts {
        let encoding = tokenizer
            .encode(*text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids = encoding.get_ids();
        if ids.len() < 2 {
            continue;
        }
        let len = ids.len().min(max_len + 1);
        let ids = &ids[..len];
        let input_ids = Tensor::new(&ids[..len - 1], device)?.unsqueeze(0)?;
        let target_ids = Tensor::new(&ids[1..len], device)?.unsqueeze(0)?;
        dataset.push((input_ids, target_ids));
    }
    println!("Dataset: {} examples, max_len={}", dataset.len(), max_len);
    Ok(dataset)
}

fn main() -> Result<()> {
    println!("=== Qwen2.5-Coder LoRA Fine-Tuning ===\n");

    // 1. Device
    // Metal autograd doesn't propagate gradients through the full model graph (candle limitation).
    // Use CPU for training. Metal works for inference.
    let device = Device::Cpu;
    println!("Device: {:?}\n", device);

    // 2. Download & load model
    let model_dir = download_qwen()?;
    let (mut model, config, tokenizer) = load_model(&model_dir, &device)?;

    // 3. Training data
    let training_texts = [
        "fn fibonacci(n: u32) -> u32 { if n <= 1 { return n; } fibonacci(n-1) + fibonacci(n-2) }",
        "fn factorial(n: u64) -> u64 { (1..=n).product() }",
        "fn is_prime(n: u64) -> bool { if n < 2 { return false; } (2..=(n as f64).sqrt() as u64).all(|i| n % i != 0) }",
        "fn gcd(a: u64, b: u64) -> u64 { if b == 0 { a } else { gcd(b, a % b) } }",
        "fn reverse_string(s: &str) -> String { s.chars().rev().collect() }",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> { let (mut lo, mut hi) = (0, arr.len()); while lo < hi { let mid = lo + (hi - lo) / 2; match arr[mid].cmp(&target) { std::cmp::Ordering::Equal => return Some(mid), std::cmp::Ordering::Less => lo = mid + 1, std::cmp::Ordering::Greater => hi = mid, } } None }",
        "fn flatten<T: Clone>(nested: &[Vec<T>]) -> Vec<T> { nested.iter().flat_map(|v| v.iter().cloned()).collect() }",
        "fn zip_with<A, B, C>(a: &[A], b: &[B], f: impl Fn(&A, &B) -> C) -> Vec<C> { a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect() }",
    ];
    let dataset = prepare_dataset(&training_texts, &tokenizer, 128, &device)?;

    // 4. Create LoRA adapter with correct GQA dimensions
    let lora_config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };

    let adapter = Arc::new(LoRAAdapter::new_with_gqa(
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        Some(config.num_kv_heads()),
        Some(config.head_dim()),
        &lora_config,
        &device,
    )?);

    println!(
        "LoRA params: {} ({:.3}% of model)",
        adapter.num_trainable_parameters(),
        100.0 * adapter.num_trainable_parameters() as f64 / model.num_parameters() as f64,
    );

    // 5. Apply adapter to model — LoRA Vars are now part of the forward pass graph
    model.apply_adapter(Arc::clone(&adapter))?;

    // 6. Collect trainable vars from the model's projections (not the adapter)
    // These are the actual Vars in the forward pass graph, so backward() finds them
    let trainable_vars: Vec<_> = model.lora_vars().into_iter().cloned().collect();
    println!("Trainable tensors: {}\n", trainable_vars.len());

    // 7. Training loop
    let num_epochs = 3;
    let total_steps = num_epochs * dataset.len();
    let scheduler = LRScheduler::WarmupCosine {
        warmup_steps: 5,
        max_lr: 2e-4,
        min_lr: 1e-6,
        total_steps,
    };
    let mut optimizer = AdamW::new(AdamWConfig::default())?;
    let mut step = 0;

    println!("Training: {num_epochs} epochs, {} batches/epoch\n", dataset.len());

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let epoch_start = std::time::Instant::now();

        for (input_ids, target_ids) in &dataset {
            // Forward
            let logits = model.forward(input_ids, None)?;

            // Loss
            let loss =
                metal_candle::training::cross_entropy_loss(&logits, target_ids, Some(u32::MAX))?;
            let loss_val = loss.to_vec0::<f32>()?;
            epoch_loss += loss_val;

            // Backward
            let grads = loss.backward()?;

            // Update
            let lr = scheduler.get_lr(step);
            optimizer.set_lr(lr);
            for var in &trainable_vars {
                if let Some(grad) = grads.get(var) {
                    optimizer.step_var(var, &grad)?;
                }
            }

            step += 1;
        }

        let elapsed = epoch_start.elapsed();
        println!(
            "Epoch {}/{}: loss={:.4}, lr={:.2e}, {:.2}s",
            epoch + 1,
            num_epochs,
            epoch_loss / dataset.len() as f32,
            scheduler.get_lr(step),
            elapsed.as_secs_f32(),
        );
    }

    // 8. Save checkpoint
    let path = "qwen_lora_finetune.safetensors";
    save_checkpoint(
        &adapter,
        path,
        Some(&CheckpointMetadata {
            global_step: step,
            loss: 0.0,
            learning_rate: 0.0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
        }),
    )?;
    println!("\nCheckpoint saved to {path}");

    Ok(())
}
