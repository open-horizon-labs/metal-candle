//! Embeddings Performance Benchmark
//!
//! Compares CPU vs Metal GPU performance for sentence embedding generation.
//!
//! Run with: `cargo run --release --example embeddings_bench --features embeddings`

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "embeddings"))]
    {
        eprintln!("This example requires the 'embeddings' feature.");
        eprintln!(
            "Run with: cargo run --release --example embeddings_bench --features embeddings"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "embeddings")]
    run_bench()
}

#[cfg(feature = "embeddings")]
#[allow(clippy::similar_names)]
fn run_bench() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::Device;
    use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    use std::time::Instant;

    println!("\n🚀 Embeddings Performance Benchmark");
    println!("═══════════════════════════════════════════════════════\n");

    let model_type = EmbeddingModelType::E5SmallV2;
    println!("Model: E5-small-v2 (dimension: {})", model_type.dimension());

    let short_texts: Vec<&str> = vec![
        "Rust is fast.",
        "Metal accelerates GPU compute.",
        "Embeddings capture semantic meaning.",
        "Transformers revolutionized NLP.",
    ];

    let long_texts: Vec<&str> = vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency, making it ideal for performance-critical applications.",
        "Apple's Metal framework provides near-direct access to the GPU, enabling high-performance graphics and compute workloads on Apple Silicon.",
        "Sentence embeddings encode text into dense vector representations that capture semantic meaning, enabling similarity search and clustering.",
        "The Transformer architecture, introduced in the Attention Is All You Need paper, uses self-attention mechanisms to process sequences in parallel.",
    ];

    let iterations = 10;
    let batch_sizes: Vec<usize> = vec![1, 2, 4];

    // ── CPU Benchmark ──────────────────────────────────────────

    println!("\n📊 CPU Benchmark");
    println!("────────────────────────────────────────────────");

    println!("  Loading model on CPU...");
    let load_start = Instant::now();
    let cpu_model = EmbeddingModel::from_pretrained(model_type, Device::Cpu)?;
    println!(
        "  Model load: {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    // Warmup
    let _ = cpu_model.encode(&short_texts[..1])?;

    bench_model(&cpu_model, &short_texts, &long_texts, &batch_sizes, iterations)?;

    // ── Metal Benchmark ────────────────────────────────────────

    println!("\n📊 Metal GPU Benchmark");
    println!("────────────────────────────────────────────────");

    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            println!("  Metal not available: {e}");
            println!("  Skipping Metal benchmark.");
            return Ok(());
        }
    };

    println!("  Loading model on Metal...");
    let load_start = Instant::now();
    let metal_model = EmbeddingModel::from_pretrained(model_type, metal_device)?;
    println!(
        "  Model load: {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    // Warmup
    let _ = metal_model.encode(&short_texts[..1])?;

    bench_model(&metal_model, &short_texts, &long_texts, &batch_sizes, iterations)?;

    // ── Head-to-head comparison ────────────────────────────────

    println!("\n🎯 Head-to-Head Comparison");
    println!("═══════════════════════════════════════════════════════");

    let compare_iters = 20;

    let _ = cpu_model.encode(&long_texts)?;
    let start = Instant::now();
    for _ in 0..compare_iters {
        let _ = cpu_model.encode(&long_texts)?;
    }
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(compare_iters);

    let _ = metal_model.encode(&long_texts)?;
    let start = Instant::now();
    for _ in 0..compare_iters {
        let _ = metal_model.encode(&long_texts)?;
    }
    let metal_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(compare_iters);

    println!("  4 long sentences × {compare_iters} iterations:");
    println!("  CPU:   {cpu_ms:8.2}ms per batch");
    println!("  Metal: {metal_ms:8.2}ms per batch");

    let speedup = cpu_ms / metal_ms;
    if speedup > 1.0 {
        println!("  Speedup: {speedup:.2}x (Metal is faster)");
    } else {
        println!(
            "  Ratio: {:.2}x (CPU is faster — batch too small for GPU overhead)",
            1.0 / speedup
        );
    }

    // ── Correctness check ──────────────────────────────────────

    println!("\n✅ Correctness Check");
    println!("────────────────────────────────────────────────");

    let cpu_emb = cpu_model.encode(&["Rust is great"])?.to_vec2::<f32>()?;
    let metal_emb = metal_model
        .encode(&["Rust is great"])?
        .to_vec2::<f32>()?;

    let dot: f32 = cpu_emb[0]
        .iter()
        .zip(&metal_emb[0])
        .map(|(a, b)| a * b)
        .sum();
    println!("  Cosine similarity (CPU vs Metal, same input): {dot:.6}");

    if dot > 0.99 {
        println!("  ✅ Embeddings match (similarity > 0.99)");
    } else {
        println!("  ⚠️  Embeddings diverge (similarity {dot:.4})");
    }

    println!("\n✨ Benchmark complete!");

    Ok(())
}

#[cfg(feature = "embeddings")]
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
fn bench_model(
    model: &metal_candle::embeddings::EmbeddingModel,
    short_texts: &[&str],
    long_texts: &[&str],
    batch_sizes: &[usize],
    iterations: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    for (label, texts) in [("short", short_texts), ("long", long_texts)] {
        for &batch_size in batch_sizes {
            let batch: Vec<&str> = texts.iter().copied().cycle().take(batch_size).collect();

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = model.encode(&batch)?;
            }
            let elapsed = start.elapsed();
            let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);
            let per_item_ms = per_iter_ms / batch_size as f64;

            println!(
                "  {label:5} batch={batch_size}: {per_iter_ms:8.2}ms total, {per_item_ms:8.2}ms/item"
            );
        }
    }

    Ok(())
}
