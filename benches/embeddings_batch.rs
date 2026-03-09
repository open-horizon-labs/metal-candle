//! Embeddings benchmarks for metal-candle.
//!
//! Measures embedding model performance across batch sizes and devices:
//! - CPU vs Metal throughput
//! - Batch size scaling
//! - Per-item latency
//!
//! Run with: `cargo bench --features embeddings --bench embeddings_batch`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "embeddings")]
use candle_core::Device;
#[cfg(feature = "embeddings")]
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

#[cfg(feature = "embeddings")]
fn benchmark_embeddings_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings_cpu");

    let model = match EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, Device::Cpu) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load CPU model: {e}");
            return;
        }
    };

    let sample_text = "Rust is a systems programming language that runs blazingly fast, \
        prevents segfaults, and guarantees thread safety.";

    for batch_size in [1, 4, 16, 64] {
        let texts: Vec<&str> = (0..batch_size).map(|_| sample_text).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("encode", format!("batch_{batch_size}")),
            &texts,
            |b, texts| {
                b.iter(|| {
                    let embeddings = model.encode(black_box(texts)).expect("Encode failed");
                    black_box(embeddings)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "embeddings")]
fn benchmark_embeddings_metal(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings_metal");

    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Metal not available: {e}");
            return;
        }
    };

    let model = match EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, metal_device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load Metal model: {e}");
            return;
        }
    };

    let sample_text = "Rust is a systems programming language that runs blazingly fast, \
        prevents segfaults, and guarantees thread safety.";

    for batch_size in [1, 4, 16, 64] {
        let texts: Vec<&str> = (0..batch_size).map(|_| sample_text).collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("encode", format!("batch_{batch_size}")),
            &texts,
            |b, texts| {
                b.iter(|| {
                    let embeddings = model.encode(black_box(texts)).expect("Encode failed");
                    black_box(embeddings)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "embeddings")]
criterion_group!(benches, benchmark_embeddings_cpu, benchmark_embeddings_metal);

#[cfg(not(feature = "embeddings"))]
fn no_embeddings(_c: &mut Criterion) {
    eprintln!("This benchmark requires the 'embeddings' feature.");
}

#[cfg(not(feature = "embeddings"))]
criterion_group!(benches, no_embeddings);

criterion_main!(benches);
