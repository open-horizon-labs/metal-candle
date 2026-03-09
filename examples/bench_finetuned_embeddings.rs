//! Benchmark: base E5 vs LoRA fine-tuned E5 on Rust retrieval.
//!
//! Trains on Rust code query-document pairs, then evaluates retrieval
//! quality (MRR@5, Recall@1) on held-out pairs.
//!
//! Run with: `cargo run --release --features embeddings --example bench_finetuned_embeddings`

use anyhow::Result;
use candle_core::Device;
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
use metal_candle::training::{contrastive_loss, AdamW, AdamWConfig, LoRAConfig, LRScheduler};

/// Compute retrieval metrics: given queries, find the best-matching document.
/// Returns (MRR@5, Recall@1, per-query ranks).
fn eval_retrieval(
    model: &EmbeddingModel,
    queries: &[&str],
    documents: &[&str],
) -> Result<(f32, f32, Vec<usize>)> {
    let q_embs = model.encode(queries)?;
    let d_embs = model.encode(documents)?;

    // Similarity matrix: (num_queries, num_docs)
    let sim = q_embs.matmul(&d_embs.t()?)?.to_vec2::<f32>()?;

    let mut mrr_sum = 0.0f32;
    let mut recall_at_1 = 0.0f32;
    let mut ranks = Vec::new();

    for (i, row) in sim.iter().enumerate() {
        // Sort document indices by similarity (descending)
        let mut scored: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // The correct document for query i is document i (diagonal)
        let rank = scored.iter().position(|(idx, _)| *idx == i).unwrap() + 1;
        ranks.push(rank);

        if rank <= 5 {
            mrr_sum += 1.0 / rank as f32;
        }
        if rank == 1 {
            recall_at_1 += 1.0;
        }
    }

    let n = queries.len() as f32;
    Ok((mrr_sum / n, recall_at_1 / n, ranks))
}

fn main() -> Result<()> {
    println!("=== E5 Base vs LoRA Fine-Tuned: Rust Retrieval Benchmark ===\n");

    let device = Device::Cpu;

    // Training pairs (Rust-specific)
    let train_queries = [
        "How to sort a vector in Rust?",
        "Rust error handling best practices",
        "What is ownership in Rust?",
        "How to create a HashMap?",
        "Async programming in Rust",
        "Pattern matching examples",
        "Trait implementations in Rust",
        "Lifetime annotations explained",
        "How to read a file in Rust?",
        "Rust iterator methods",
        "How to parse JSON in Rust?",
        "Concurrency with threads in Rust",
    ];
    let train_docs = [
        "Use vec.sort() or vec.sort_by() for custom ordering. sort_unstable() is faster.",
        "Use Result<T, E> with the ? operator. Define custom error types with thiserror crate.",
        "Ownership is Rust's memory management system. Each value has one owner. When owner goes out of scope, value is dropped.",
        "Use std::collections::HashMap::new() or collect from iterator of tuples.",
        "Use async/await with tokio runtime. Futures are lazy and need an executor.",
        "Match on enums, use if let for single patterns, while let for loops. Guards with if.",
        "impl TraitName for Type { fn method(&self) {} }. Use derive for common traits.",
        "'a annotations tell the compiler how references relate. Elision rules handle most cases.",
        "Use std::fs::read_to_string() for small files, BufReader for large files.",
        "map, filter, fold, collect, enumerate, zip, chain, flat_map are essential.",
        "Use serde and serde_json. Derive Serialize/Deserialize on structs.",
        "Use std::thread::spawn or rayon for parallelism. Arc<Mutex<T>> for shared state.",
    ];

    // Held-out evaluation pairs (different but related queries)
    let eval_queries = [
        "sorting elements in a vec",
        "handling errors with Result",
        "memory management without garbage collector",
        "key-value store data structure",
        "non-blocking I/O in Rust",
        "destructuring in match expressions",
        "implementing Display trait",
        "borrow checker lifetime rules",
        "file I/O operations",
        "functional programming with iterators",
        "deserializing JSON strings",
        "parallel execution with multiple threads",
    ];
    // Same documents — eval queries should match the same docs
    let eval_docs = &train_docs;

    // 1. Baseline: evaluate base model
    println!("Loading E5-small-v2...");
    let mut model =
        EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device.clone())?;

    let (base_mrr, base_r1, base_ranks) = eval_retrieval(&model, &eval_queries, eval_docs)?;
    println!("BASE MODEL:");
    println!("  MRR@5:     {base_mrr:.4}");
    println!("  Recall@1:  {base_r1:.4}");
    println!("  Ranks:     {base_ranks:?}\n");

    // 2. Fine-tune
    println!("Fine-tuning with LoRA (rank=8, 10 epochs)...");
    let lora_config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    };
    model.apply_lora(&lora_config)?;
    let trainable_vars: Vec<candle_core::Var> =
        model.lora_vars().into_iter().cloned().collect();
    println!("  LoRA params: {}", trainable_vars.iter().map(|v| v.as_tensor().elem_count()).sum::<usize>());

    let num_epochs = 10;
    let scheduler = LRScheduler::WarmupCosine {
        warmup_steps: 3,
        max_lr: 2e-4,
        min_lr: 1e-6,
        total_steps: num_epochs,
    };
    let mut optimizer = AdamW::new(AdamWConfig::default())?;

    let train_q: Vec<&str> = train_queries.to_vec();
    let train_d: Vec<&str> = train_docs.to_vec();

    for epoch in 0..num_epochs {
        let q_embs = model.encode(&train_q)?;
        let d_embs = model.encode(&train_d)?;
        let loss = contrastive_loss(&q_embs, &d_embs, 20.0)?;
        let loss_val = loss.to_vec0::<f32>()?;
        let grads = loss.backward()?;

        let lr = scheduler.get_lr(epoch);
        optimizer.set_lr(lr);
        for var in &trainable_vars {
            if let Some(grad) = grads.get(var) {
                optimizer.step_var(var, &grad)?;
            }
        }

        if epoch % 2 == 0 || epoch == num_epochs - 1 {
            println!("  Epoch {}/{}: loss={loss_val:.4}, lr={lr:.2e}", epoch + 1, num_epochs);
        }
    }

    // 3. Evaluate fine-tuned model
    let (ft_mrr, ft_r1, ft_ranks) = eval_retrieval(&model, &eval_queries, eval_docs)?;
    println!("\nFINE-TUNED MODEL:");
    println!("  MRR@5:     {ft_mrr:.4}");
    println!("  Recall@1:  {ft_r1:.4}");
    println!("  Ranks:     {ft_ranks:?}");

    // 4. Comparison
    println!("\n=== COMPARISON ===");
    println!("              Base    Fine-tuned   Delta");
    println!("  MRR@5:     {base_mrr:.4}    {ft_mrr:.4}      {:+.4}", ft_mrr - base_mrr);
    println!("  Recall@1:  {base_r1:.4}    {ft_r1:.4}      {:+.4}", ft_r1 - base_r1);

    let base_correct = base_ranks.iter().filter(|&&r| r == 1).count();
    let ft_correct = ft_ranks.iter().filter(|&&r| r == 1).count();
    println!(
        "  Correct@1: {}/{}     {}/{}",
        base_correct,
        eval_queries.len(),
        ft_correct,
        eval_queries.len()
    );

    // Per-query breakdown for queries that changed
    println!("\n  Per-query rank changes:");
    for (i, q) in eval_queries.iter().enumerate() {
        if base_ranks[i] != ft_ranks[i] {
            let arrow = if ft_ranks[i] < base_ranks[i] { "improved" } else { "regressed" };
            println!("    '{}': {} -> {} ({})", q, base_ranks[i], ft_ranks[i], arrow);
        }
    }

    // 5. Save if improved
    if ft_mrr > base_mrr {
        let lora_path = std::path::Path::new("e5_rust_lora.safetensors");
        let tensors: std::collections::HashMap<String, candle_core::Tensor> = trainable_vars
            .iter()
            .enumerate()
            .map(|(i, var)| (format!("lora.{i}"), var.as_tensor().clone()))
            .collect();
        candle_core::safetensors::save(&tensors, lora_path)?;
        println!("\nLoRA weights saved to {} (ready for HF Hub upload)", lora_path.display());
    } else {
        println!("\nNo improvement — skipping save.");
    }

    Ok(())
}
