//! SIMD cosine similarity and MMR reranking (Phase 2).
#![cfg_attr(feature = "nightly-simd", feature(portable_simd))]

use m3_error::{M3Error, Result};
use rayon::prelude::*;

/// Reinterpret a SQLite BLOB (`&[u8]`) as `&[f32]` with no copy.
///
/// Replaces Python's `struct.unpack`. Errors if the byte length is not a
/// multiple of 4 or the buffer is not 4-byte aligned for `f32`.
pub fn blob_as_f32(blob: &[u8]) -> Result<&[f32]> {
    bytemuck::try_cast_slice(blob).map_err(|e| M3Error::Other(format!("blob cast failed: {e}")))
}

/// Reinterpret an `&[f32]` embedding back into its raw byte representation.
pub fn f32_as_blob(vec: &[f32]) -> &[u8] {
    bytemuck::cast_slice(vec)
}

/// Cosine similarity between two equal-length vectors (scalar fallback).
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(M3Error::VectorDimMismatch { expected: a.len(), got: b.len() });
    }
    Ok(cosine_unchecked(a, b))
}

/// Cosine without the length check; auto-vectorizable chunked accumulation.
fn cosine_unchecked(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut dot = [0.0f32; LANES];
    let mut na = [0.0f32; LANES];
    let mut nb = [0.0f32; LANES];

    let chunks = a.len() / LANES;
    for c in 0..chunks {
        let base = c * LANES;
        for l in 0..LANES {
            let x = a[base + l];
            let y = b[base + l];
            dot[l] += x * y;
            na[l] += x * x;
            nb[l] += y * y;
        }
    }
    let mut d = 0.0f32;
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    for l in 0..LANES {
        d += dot[l];
        sa += na[l];
        sb += nb[l];
    }
    for i in (chunks * LANES)..a.len() {
        d += a[i] * b[i];
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    if sa == 0.0 || sb == 0.0 {
        return 0.0;
    }
    let c = d / (sa.sqrt() * sb.sqrt());
    c.clamp(-1.0, 1.0)
}

/// Score `query` against every vector in `corpus`, in parallel.
pub fn cosine_batch(query: &[f32], corpus: &[&[f32]]) -> Result<Vec<f32>> {
    for (i, v) in corpus.iter().enumerate() {
        if v.len() != query.len() {
            return Err(M3Error::VectorDimMismatch { expected: query.len(), got: v.len() });
        }
        let _ = i;
    }
    Ok(corpus.par_iter().map(|v| cosine_unchecked(query, v)).collect())
}

/// Maximal Marginal Relevance rerank over candidate vectors.
///
/// Selects `min(k, candidates.len())` indices balancing relevance to `query`
/// (weight `lambda`) against diversity from already-selected items
/// (weight `1 - lambda`). Returns selected indices in selection order.
pub fn mmr_rerank(
    query: &[f32],
    candidates: &[&[f32]],
    lambda: f32,
    k: usize,
) -> Result<Vec<usize>> {
    let query_sim = cosine_batch(query, candidates)?;
    let n = candidates.len();
    let take = k.min(n);
    let mut selected: Vec<usize> = Vec::with_capacity(take);
    let mut remaining: Vec<usize> = (0..n).collect();

    while selected.len() < take {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (pos, &cand) in remaining.iter().enumerate() {
            let max_div = selected
                .iter()
                .map(|&s| cosine_unchecked(candidates[cand], candidates[s]))
                .fold(0.0f32, f32::max);
            let score = lambda * query_sim[cand] - (1.0 - lambda) * max_div;
            if score > best_score {
                best_score = score;
                best_idx = pos;
            }
        }
        selected.push(remaining.swap_remove(best_idx));
    }
    Ok(selected)
}
