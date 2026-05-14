//! SIMD cosine similarity and MMR reranking (Phase 2).
#![cfg_attr(feature = "nightly-simd", feature(portable_simd))]

use m3_error::{M3Error, Result};
use rayon::prelude::*;

mod displacement;
pub use displacement::{
    displacement_permutation, enforce_displacement_guard, DisplacementRow,
};

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

/// Policy-aware MMR that mirrors m3-memory's retrieval-ranking loop.
///
/// Unlike [`mmr_rerank`], the relevance term is a caller-supplied score per
/// candidate (a blended FTS+vector rank score), not cosine-to-query. The
/// diversity term is still cosine between candidate vectors.
///
/// `relevance[i]` and `candidate_vectors[i]` describe the same candidate.
/// When `force_seed_first` is true, index 0 is selected first unconditionally
/// (caller must pre-sort descending by relevance); the rest are greedily picked
/// by `lambda * relevance[i] - (1 - lambda) * max_cosine_to_selected`. When
/// false, selection is pure greedy from an empty `selected` set.
///
/// `max_sim` is 0.0 when `selected` is empty. Returns `min(k, n)` indices in
/// selection order.
pub fn mmr_rerank_scored(
    relevance: &[f32],
    candidate_vectors: &[&[f32]],
    lambda: f32,
    k: usize,
    force_seed_first: bool,
) -> Result<Vec<usize>> {
    let n = relevance.len();
    if candidate_vectors.len() != n {
        return Err(M3Error::Other(format!(
            "relevance/vector length mismatch: {} vs {}",
            n,
            candidate_vectors.len()
        )));
    }
    let take = k.min(n);
    let mut selected: Vec<usize> = Vec::with_capacity(take);
    let mut remaining: Vec<usize> = (0..n).collect();
    if take == 0 {
        return Ok(selected);
    }

    if force_seed_first {
        selected.push(remaining.remove(0));
    }

    while selected.len() < take {
        let mut best_pos = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (pos, &cand) in remaining.iter().enumerate() {
            let max_sim = selected
                .iter()
                .map(|&s| cosine_unchecked(candidate_vectors[cand], candidate_vectors[s]))
                .fold(0.0f32, f32::max);
            let score = lambda * relevance[cand] - (1.0 - lambda) * max_sim;
            if score > best_score {
                best_score = score;
                best_pos = pos;
            }
        }
        selected.push(remaining.remove(best_pos));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_force_seed_picks_index_zero_first() {
        // index 1 has higher relevance, but force_seed_first pins index 0.
        let v0: &[f32] = &[1.0, 0.0];
        let v1: &[f32] = &[0.0, 1.0];
        let v2: &[f32] = &[1.0, 1.0];
        let rel = [0.5, 0.9, 0.1];
        let cands = [v0, v1, v2];
        let out = mmr_rerank_scored(&rel, &cands, 0.7, 3, true).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn scored_no_seed_is_pure_greedy() {
        let v0: &[f32] = &[1.0, 0.0];
        let v1: &[f32] = &[0.0, 1.0];
        let rel = [0.5, 0.9];
        let cands = [v0, v1];
        let out = mmr_rerank_scored(&rel, &cands, 0.7, 2, false).unwrap();
        assert_eq!(out, vec![1, 0]);
    }

    #[test]
    fn scored_k_greater_than_n() {
        let v0: &[f32] = &[1.0, 0.0];
        let rel = [0.5];
        let cands = [v0];
        assert_eq!(mmr_rerank_scored(&rel, &cands, 0.7, 9, true).unwrap(), vec![0]);
    }
}
