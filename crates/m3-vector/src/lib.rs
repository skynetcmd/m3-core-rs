//! SIMD cosine similarity and MMR reranking (Phase 2).

use m3_error::{M3Error, Result};
use rayon::prelude::*;
use wide::f32x8;

mod displacement;
pub mod index;
pub use displacement::{displacement_permutation, enforce_displacement_guard, DisplacementRow};
pub use index::VectorIndex;

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
        return Err(M3Error::VectorDimMismatch {
            expected: a.len(),
            got: b.len(),
            index: None,
        });
    }
    Ok(cosine_unchecked(a, b))
}

/// Cosine without the length check; SIMD-accumulated via the `wide` crate
/// (AVX2/SSE2/NEON where available, scalar fallback otherwise).
fn cosine_unchecked(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    let mut dot = f32x8::ZERO;
    let mut na = f32x8::ZERO;
    let mut nb = f32x8::ZERO;
    for c in 0..chunks {
        let base = c * LANES;
        // Loading via fixed-size array gives the compiler a known-length
        // contiguous load (cheaper than the `From<&[f32]>` match-on-length).
        let av_arr: [f32; LANES] = a[base..base + LANES].try_into().unwrap();
        let bv_arr: [f32; LANES] = b[base..base + LANES].try_into().unwrap();
        let av = f32x8::new(av_arr);
        let bv = f32x8::new(bv_arr);
        dot = av.mul_add(bv, dot);
        na = av.mul_add(av, na);
        nb = bv.mul_add(bv, nb);
    }
    let mut d = dot.reduce_add();
    let mut sa = na.reduce_add();
    let mut sb = nb.reduce_add();
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
            return Err(M3Error::VectorDimMismatch {
                expected: query.len(),
                got: v.len(),
                index: Some(i),
            });
        }
    }
    Ok(corpus
        .par_iter()
        .map(|v| cosine_unchecked(query, v))
        .collect())
}

/// Decode one packed row of little-endian f32s: borrowed when the bytes are
/// 4-byte aligned (the usual case), copied otherwise. Alignment is a property
/// of the caller's allocator, not of the data, so it must never change a score
/// or fail a call. `b.len()` must be a multiple of 4 (callers check the length
/// first); a trailing partial float is ignored.
pub fn decode_packed_row(b: &[u8]) -> std::borrow::Cow<'_, [f32]> {
    match bytemuck::try_cast_slice::<u8, f32>(b) {
        Ok(v) => std::borrow::Cow::Borrowed(v),
        Err(_) => std::borrow::Cow::Owned(
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ),
    }
}

/// Warn once per call about rows that are not `dim * 4` bytes (they score 0.0).
/// The count and the first offender are what an operator needs to find the
/// truncated or wrong-model blob; silently zero-scoring hid them.
fn warn_wrong_length_rows(op: &str, blobs: &[&[u8]], dim: usize) {
    let expected = dim * 4;
    let mut bad = blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| b.len() != expected);
    if let Some((first, b)) = bad.next() {
        log::warn!(
            "{op}: {} of {} candidate rows are not {expected} bytes (dim {dim} x 4) and scored 0.0; \
             first is row {first} with {} bytes — a truncated blob or one written by a model of \
             another dimension",
            1 + bad.count(),
            blobs.len(),
            b.len(),
        );
    }
}

/// Score `query` against every packed-blob embedding in `blobs`, in parallel.
///
/// Each blob must be exactly `dim * 4` bytes (a sequence of little-endian f32s
/// — the on-disk layout produced by `f32_as_blob`). A blob of any other length
/// scores 0.0 for that row rather than erroring — parity with the Python
/// fallback on ragged inputs, so a single corrupt row cannot kill a whole
/// retrieval — and is reported with `log::warn!`.
///
/// This is the load-bearing primitive for the retrieval hot path: the caller
/// passes raw SQLite BLOB bytes straight in, avoiding a per-row `struct.unpack`
/// + Python `list` allocation and a `Vec<Vec<f32>>` FFI marshalling step.
pub fn cosine_batch_packed(query: &[f32], blobs: &[&[u8]], dim: usize) -> Result<Vec<f32>> {
    if query.len() != dim {
        return Err(M3Error::VectorDimMismatch {
            expected: dim,
            got: query.len(),
            index: None,
        });
    }
    warn_wrong_length_rows("cosine_batch_packed", blobs, dim);
    let expected_bytes = dim * 4;
    Ok(blobs
        .par_iter()
        .map(|b| {
            if b.len() != expected_bytes {
                return 0.0;
            }
            cosine_unchecked(query, &decode_packed_row(b))
        })
        .collect())
}

/// Max-pooled multi-anchor cosine over packed candidate blobs.
///
/// For each candidate blob, returns `max_j cosine(anchors[j], candidate)` — the
/// "max-similarity" rerank signal: scoring a candidate by its closeness to the
/// BEST-matching anchor (rather than to a single averaged/centroid anchor) preserves
/// sharp topical matches that a centroid washes out. Pushes the whole
/// N-candidates × M-anchors matrix op into Rust so the Python caller never assembles
/// it (avoids a per-anchor FFI loop). Data-parallel over candidates (rayon); each
/// candidate scans all anchors with the SIMD kernel.
///
/// `anchors` is a slice of M anchor vectors, each length `dim`. `blobs` are the
/// candidate embeddings as raw little-endian f32 bytes (`dim * 4` each). A blob of
/// the wrong byte length scores 0.0 and is warned about (as in `cosine_batch_packed`).
/// An empty anchor list yields all -1.0 (no signal — caller treats as "unscored",
/// mirroring the pure-Python `max(... for a in anchors)` over an empty set being
/// guarded upstream). Errors if any anchor's length != dim.
pub fn cosine_batch_maxpool_packed(
    anchors: &[&[f32]],
    blobs: &[&[u8]],
    dim: usize,
) -> Result<Vec<f32>> {
    for (j, a) in anchors.iter().enumerate() {
        if a.len() != dim {
            return Err(M3Error::VectorDimMismatch {
                expected: dim,
                got: a.len(),
                index: Some(j),
            });
        }
    }
    warn_wrong_length_rows("cosine_batch_maxpool_packed", blobs, dim);
    let expected_bytes = dim * 4;
    Ok(blobs
        .par_iter()
        .map(|b| {
            if b.len() != expected_bytes {
                return 0.0;
            }
            let v = decode_packed_row(b);
            // max cosine over all anchors; -1.0 when there are no anchors (cosine is
            // in [-1, 1], so -1.0 is the correct identity for an empty max).
            anchors
                .iter()
                .map(|a| cosine_unchecked(a, &v))
                .fold(-1.0_f32, f32::max)
        })
        .collect())
}

/// Vectorized per-row hybrid score combining cosine, BM25, length penalty,
/// title-overlap, and importance — the body of the per-row Python loop in
/// `memory_search_scored_impl`, fully data-parallel.
///
/// All input slices must have identical length `N` (one entry per candidate).
/// Returns a `Vec<f32>` of length `N` with the same blended score the Python
/// path computes. Caller still adds role/intent boosts and temporal/recency
/// adjustments downstream — those are sparse or query-conditional and don't
/// vectorize cleanly.
///
/// Formula per row `i`:
///   raw      = vector_scores[i] * vector_weight + (1/(1+|bm25_scores[i]|)) * (1 - vector_weight)
///   penalty  = if content_lens[i] < short_turn_threshold
///                then max(0.3, content_lens[i] / short_turn_threshold)
///                else 1.0
///   final[i] = raw * penalty
///            + title_match_boost * title_overlaps[i]
///            + importance_weight * importances[i]
#[allow(clippy::too_many_arguments)]
pub fn hybrid_score_batch(
    vector_scores: &[f32],
    bm25_scores: &[f32],
    content_lens: &[u32],
    importances: &[f32],
    title_overlaps: &[f32],
    vector_weight: f32,
    importance_weight: f32,
    title_match_boost: f32,
    short_turn_threshold: u32,
) -> Result<Vec<f32>> {
    let n = vector_scores.len();
    if bm25_scores.len() != n
        || content_lens.len() != n
        || importances.len() != n
        || title_overlaps.len() != n
    {
        return Err(M3Error::Config(format!(
            "hybrid_score_batch length mismatch: vec={}, bm25={}, lens={}, imp={}, titles={}",
            n,
            bm25_scores.len(),
            content_lens.len(),
            importances.len(),
            title_overlaps.len(),
        )));
    }
    let stt = short_turn_threshold.max(1) as f32;
    let inv_stt = 1.0 / stt;
    let stt_threshold = short_turn_threshold;
    let bm25_weight = 1.0 - vector_weight;

    // Below ~256 rows, rayon overhead exceeds the gain on this body; go
    // sequential. Above the threshold, use the parallel index-mapped path
    // (rayon doesn't have a clean 5-way `par_iter` zip).
    if n < 256 {
        Ok(vector_scores
            .iter()
            .zip(bm25_scores.iter())
            .zip(content_lens.iter())
            .zip(importances.iter())
            .zip(title_overlaps.iter())
            .map(|((((vs, bm), cl), imp), title)| {
                let bm25_norm = 1.0 / (1.0 + bm.abs());
                let raw = vs * vector_weight + bm25_norm * bm25_weight;
                let penalty = if *cl < stt_threshold {
                    ((*cl as f32) * inv_stt).max(0.3)
                } else {
                    1.0
                };
                raw * penalty + title_match_boost * title + importance_weight * imp
            })
            .collect())
    } else {
        Ok((0..n)
            .into_par_iter()
            .map(|i| {
                let bm25_norm = 1.0 / (1.0 + bm25_scores[i].abs());
                let raw = vector_scores[i] * vector_weight + bm25_norm * bm25_weight;
                let cl = content_lens[i];
                let penalty = if cl < stt_threshold {
                    ((cl as f32) * inv_stt).max(0.3)
                } else {
                    1.0
                };
                raw * penalty
                    + title_match_boost * title_overlaps[i]
                    + importance_weight * importances[i]
            })
            .collect())
    }
}

/// Rank-based linear recency bonus aligned to `valid_froms`.
///
/// Items with `Some(non-empty)` `valid_from` are ordered by their numeric
/// `YYYY MM DD HH MM SS` fields (see `iso_to_u64`), which matches a string sort
/// of well-formed ISO-8601. ⚠ A timezone offset is NOT applied: `10:00+05:00`
/// and `10:00Z` compare equal, so mixed-offset inputs can be mis-ordered by up
/// to the offset difference. The oldest dated item gets bonus 0, the newest gets
/// `bias`, with linear interpolation between. Items with `None` or empty strings
/// get bonus 0. When fewer than two dated items exist, returns all zeros.
///
/// Returned vector aligns 1:1 with the input — apply by adding to the existing
/// hybrid score.
pub fn recency_bonus_ranks(valid_froms: &[Option<String>], bias: f32) -> Vec<f32> {
    let n = valid_froms.len();
    let mut out = vec![0.0f32; n];
    if bias <= 0.0 || n < 2 {
        return out;
    }
    // Parse ISO-8601 to a packed u64 once per row; integer compare is much
    // cheaper than lexicographic string compare. Accepts "YYYY-MM-DD" and
    // "YYYY-MM-DDTHH:MM:SS..." forms; missing parts default to 0. Unparseable
    // strings collapse to 0 (treated as oldest), matching the prior string
    // sort's behavior on non-ISO inputs. Arithmetic saturates: an over-long
    // digit run (or a year past ~1.8e9) sorts as newest instead of wrapping or
    // panicking on caller data.
    fn iso_to_u64(s: &str) -> u64 {
        let mut parts = [0u64; 6]; // year, month, day, hour, min, sec
        let mut idx = 0usize;
        let mut current = 0u64;
        let mut digits = 0u32;
        for ch in s.bytes() {
            if ch.is_ascii_digit() {
                current = current
                    .saturating_mul(10)
                    .saturating_add((ch - b'0') as u64);
                digits += 1;
            } else if digits > 0 {
                if idx < 6 {
                    parts[idx] = current;
                    idx += 1;
                }
                current = 0;
                digits = 0;
                if idx >= 6 {
                    break;
                }
            }
        }
        if digits > 0 && idx < 6 {
            parts[idx] = current;
        }
        [10_000_000_000u64, 100_000_000, 1_000_000, 10_000, 100, 1]
            .iter()
            .zip(parts)
            .fold(0u64, |acc, (scale, p)| {
                acc.saturating_add(p.saturating_mul(*scale))
            })
    }

    let mut dated: Vec<(usize, u64)> = valid_froms
        .iter()
        .enumerate()
        .filter_map(|(i, v)| match v {
            Some(s) if !s.is_empty() => Some((i, iso_to_u64(s))),
            _ => None,
        })
        .collect();
    if dated.len() < 2 {
        return out;
    }
    dated.sort_unstable_by_key(|&(_, k)| k);
    let denom = (dated.len() - 1) as f32;
    for (rank, (orig_idx, _)) in dated.into_iter().enumerate() {
        out[orig_idx] = bias * (rank as f32) / denom;
    }
    out
}

/// Candidate count at which an MMR step scores candidates with rayon.
const MMR_PAR_THRESHOLD: usize = 128;

/// The better of two `(position, score)` pairs, one total order shared by the
/// sequential and the rayon branch of each MMR step so they cannot disagree:
/// the higher score wins, a tie goes to the lower position, and a NaN score
/// ranks as -inf (it never beats a number). The rayon branch used
/// `partial_cmp().unwrap()`, which panicked on NaN — only above 128 candidates,
/// while the sequential branch quietly kept position 0.
fn mmr_better(a: (usize, f32), b: (usize, f32)) -> (usize, f32) {
    let key = |s: f32| if s.is_nan() { f32::NEG_INFINITY } else { s };
    let (ka, kb) = (key(a.1), key(b.1));
    if kb > ka || (kb == ka && b.0 < a.0) {
        b
    } else {
        a
    }
}

/// Position in `remaining` of the best-scoring candidate; `None` when empty.
fn mmr_pick(remaining: &[usize], score: impl Fn(usize) -> f32 + Sync) -> Option<usize> {
    let scored = |(pos, &cand): (usize, &usize)| (pos, score(cand));
    let best = if remaining.len() < MMR_PAR_THRESHOLD {
        remaining.iter().enumerate().map(scored).reduce(mmr_better)
    } else {
        remaining
            .par_iter()
            .enumerate()
            .map(scored)
            .reduce_with(mmr_better)
    };
    best.map(|(pos, _)| pos)
}

/// Warn when an MMR input carries NaN/inf: those candidates are ranked last
/// rather than failing the search, but the cause must be visible.
fn warn_non_finite(op: &str, what: &str, values: &[f32]) {
    let mut bad = values.iter().enumerate().filter(|(_, v)| !v.is_finite());
    if let Some((first, v)) = bad.next() {
        log::warn!(
            "{op}: {} of {} {what} values are not finite (first: index {first} = {v}); those \
             candidates rank last — check the embedding blobs / scores that produced them",
            1 + bad.count(),
            values.len(),
        );
    }
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

    warn_non_finite("mmr_rerank", "query cosine", &query_sim);
    while selected.len() < take {
        let Some(best_pos) = mmr_pick(&remaining, |cand| {
            let max_div = selected
                .iter()
                .map(|&s| cosine_unchecked(candidates[cand], candidates[s]))
                .fold(0.0f32, f32::max);
            lambda * query_sim[cand] - (1.0 - lambda) * max_div
        }) else {
            break;
        };
        selected.push(remaining.swap_remove(best_pos));
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
        return Err(M3Error::Config(format!(
            "mmr_rerank_scored length mismatch: {} relevance scores vs {} candidate vectors",
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

    warn_non_finite("mmr_rerank_scored", "relevance", relevance);
    if force_seed_first {
        selected.push(remaining.remove(0));
    }

    while selected.len() < take {
        let Some(best_pos) = mmr_pick(&remaining, |cand| {
            let max_sim = selected
                .iter()
                .map(|&s| cosine_unchecked(candidate_vectors[cand], candidate_vectors[s]))
                .fold(0.0f32, f32::max);
            lambda * relevance[cand] - (1.0 - lambda) * max_sim
        }) else {
            break;
        };
        selected.push(remaining.remove(best_pos));
    }
    Ok(selected)
}

/// Token-set Jaccard similarity, lowercased, punctuation-stripped.
///
/// Strips anything that is not alphanumeric, whitespace, or underscore
/// (approximate parity with Python's `[^\w\s]` regex).
pub fn token_jaccard(a: &str, b: &str) -> f32 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let intersection = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    intersection / union
}

/// Batch token-set Jaccard similarity.
///
/// Returns a vector of scores, one per candidate, in parallel via rayon.
pub fn token_jaccard_batch(query: &str, candidates: &[&str]) -> Vec<f32> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return vec![0.0; candidates.len()];
    }
    candidates
        .par_iter()
        .map(|c| {
            let c_tokens = tokenize(c);
            if c_tokens.is_empty() {
                return 0.0;
            }
            let intersection = q_tokens.intersection(&c_tokens).count() as f32;
            let union = q_tokens.union(&c_tokens).count() as f32;
            intersection / union
        })
        .collect()
}

/// Combined content-dedup and MMR reranking.
///
/// 1. Sorts all candidates by relevance score descending.
/// 2. Iterates and dedups by `contents[i].trim()`.
/// 3. Runs MMR on the top `k*3` unique candidates.
/// 4. Returns the top `k` selected indices in original index order.
pub fn rank_hybrid(
    relevance: &[f32],
    contents: &[&str],
    embeddings: &[&[f32]],
    lambda: f32,
    k: usize,
) -> Result<Vec<usize>> {
    let n = relevance.len();
    if contents.len() != n || embeddings.len() != n {
        return Err(M3Error::Config(format!(
            "rank_hybrid length mismatch: relevance={}, contents={}, embeddings={}",
            n,
            contents.len(),
            embeddings.len()
        )));
    }

    // 1. Sort indices by relevance
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        relevance[b]
            .partial_cmp(&relevance[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 2. Content dedup
    let mut seen_content = std::collections::HashSet::new();
    let mut pre_ranked_idx = Vec::new();
    for idx in indices {
        let c = contents[idx].trim();
        if !c.is_empty() {
            if seen_content.contains(c) {
                continue;
            }
            seen_content.insert(c);
        }
        pre_ranked_idx.push(idx);
        if pre_ranked_idx.len() >= k.saturating_mul(3) {
            break;
        }
    }

    if pre_ranked_idx.is_empty() {
        return Ok(Vec::new());
    }

    // 3. MMR
    let mmr_relevance: Vec<f32> = pre_ranked_idx.iter().map(|&i| relevance[i]).collect();
    let mmr_vecs: Vec<&[f32]> = pre_ranked_idx.iter().map(|&i| embeddings[i]).collect();

    let sel_mmr_idx = mmr_rerank_scored(&mmr_relevance, &mmr_vecs, lambda, k, true)?;

    // 4. Map back to original indices
    Ok(sel_mmr_idx.iter().map(|&i| pre_ranked_idx[i]).collect())
}

fn tokenize(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(|t| t.to_string())
        .collect()
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
        assert_eq!(
            mmr_rerank_scored(&rel, &cands, 0.7, 9, true).unwrap(),
            vec![0]
        );
    }

    #[test]
    fn cosine_batch_packed_matches_unchecked() {
        let q: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let v0: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let v1: &[f32] = &[0.0, 1.0, 0.0, 0.0];
        let v2: &[f32] = &[0.5, 0.5, 0.0, 0.0];
        let b0 = f32_as_blob(v0).to_vec();
        let b1 = f32_as_blob(v1).to_vec();
        let b2 = f32_as_blob(v2).to_vec();
        let blobs: Vec<&[u8]> = vec![&b0, &b1, &b2];
        let got = cosine_batch_packed(q, &blobs, 4).unwrap();
        let want = [
            cosine_unchecked(q, v0),
            cosine_unchecked(q, v1),
            cosine_unchecked(q, v2),
        ];
        assert_eq!(got.len(), 3);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "got={g}, want={w}");
        }
    }

    #[test]
    fn cosine_batch_packed_ragged_zero_fills() {
        let q: &[f32] = &[1.0, 0.0];
        let v0: &[f32] = &[1.0, 0.0];
        let b_good = f32_as_blob(v0).to_vec();
        let b_short: Vec<u8> = vec![0u8; 7]; // not multiple of 4 * 2
        let blobs: Vec<&[u8]> = vec![&b_good, &b_short];
        let got = cosine_batch_packed(q, &blobs, 2).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-6);
        assert_eq!(got[1], 0.0);
    }

    #[test]
    fn cosine_batch_packed_query_dim_mismatch_errors() {
        let q: &[f32] = &[1.0, 0.0, 0.0];
        let blobs: Vec<&[u8]> = vec![];
        assert!(cosine_batch_packed(q, &blobs, 4).is_err());
    }

    #[test]
    fn cosine_batch_maxpool_takes_best_anchor() {
        // a0 aligns with v1, a1 aligns with v0 -> each candidate's score is the MAX,
        // i.e. 1.0 for both, and the off-anchor cosine must NOT win.
        let a0: &[f32] = &[0.0, 1.0, 0.0, 0.0];
        let a1: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let v0: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let v1: &[f32] = &[0.0, 1.0, 0.0, 0.0];
        let v2: &[f32] = &[0.5, 0.5, 0.0, 0.0]; // cos to a0 == cos to a1 == ~0.707
        let anchors: Vec<&[f32]> = vec![a0, a1];
        let b0 = f32_as_blob(v0).to_vec();
        let b1 = f32_as_blob(v1).to_vec();
        let b2 = f32_as_blob(v2).to_vec();
        let blobs: Vec<&[u8]> = vec![&b0, &b1, &b2];
        let got = cosine_batch_maxpool_packed(&anchors, &blobs, 4).unwrap();
        // reference: max over anchors of cosine_unchecked
        for (k, v) in [v0, v1, v2].iter().enumerate() {
            let want = anchors
                .iter()
                .map(|a| cosine_unchecked(a, v))
                .fold(-1.0_f32, f32::max);
            assert!(
                (got[k] - want).abs() < 1e-6,
                "row {k}: got={}, want={want}",
                got[k]
            );
        }
        assert!((got[0] - 1.0).abs() < 1e-6); // v0 best-matches a1
        assert!((got[1] - 1.0).abs() < 1e-6); // v1 best-matches a0
    }

    #[test]
    fn cosine_batch_maxpool_empty_anchors_is_neg_one() {
        let v0: &[f32] = &[1.0, 0.0];
        let b0 = f32_as_blob(v0).to_vec();
        let anchors: Vec<&[f32]> = vec![];
        let blobs: Vec<&[u8]> = vec![&b0];
        let got = cosine_batch_maxpool_packed(&anchors, &blobs, 2).unwrap();
        assert_eq!(got, vec![-1.0]); // no anchors -> no signal
    }

    #[test]
    fn cosine_batch_maxpool_ragged_candidate_zero_fills() {
        let a0: &[f32] = &[1.0, 0.0];
        let v0: &[f32] = &[1.0, 0.0];
        let anchors: Vec<&[f32]> = vec![a0];
        let b_good = f32_as_blob(v0).to_vec();
        let b_short: Vec<u8> = vec![0u8; 7]; // not multiple of 4*2
        let blobs: Vec<&[u8]> = vec![&b_good, &b_short];
        let got = cosine_batch_maxpool_packed(&anchors, &blobs, 2).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-6);
        assert_eq!(got[1], 0.0); // bad candidate -> 0.0, same as cosine_batch_packed
    }

    #[test]
    fn cosine_batch_maxpool_anchor_dim_mismatch_errors() {
        let a0: &[f32] = &[1.0, 0.0, 0.0]; // dim 3, but we pass dim=2
        let anchors: Vec<&[f32]> = vec![a0];
        let blobs: Vec<&[u8]> = vec![];
        assert!(cosine_batch_maxpool_packed(&anchors, &blobs, 2).is_err());
    }

    #[test]
    fn hybrid_score_batch_parity_with_python_formula() {
        let vec_s = vec![0.9f32, 0.5];
        let bm = vec![1.0f32, 4.0];
        let lens = vec![100u32, 10];
        let imp = vec![0.0f32, 1.0];
        let titles = vec![0.5f32, 0.0];
        // vw=0.7, iw=0.1, tmb=0.15, stt=40
        let out =
            hybrid_score_batch(&vec_s, &bm, &lens, &imp, &titles, 0.7, 0.1, 0.15, 40).unwrap();
        // Row 0: bm25_norm = 1/(1+1) = 0.5; raw = 0.9*0.7 + 0.5*0.3 = 0.63+0.15 = 0.78
        //        len 100 >= 40 -> penalty 1.0; final = 0.78 + 0.15*0.5 + 0.1*0.0 = 0.855
        // Row 1: bm25_norm = 1/(1+4) = 0.2; raw = 0.5*0.7 + 0.2*0.3 = 0.35+0.06 = 0.41
        //        len 10 < 40 -> penalty max(0.3, 10/40) = max(0.3, 0.25) = 0.3
        //        final = 0.41*0.3 + 0.15*0 + 0.1*1 = 0.123 + 0.1 = 0.223
        assert!((out[0] - 0.855).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 0.223).abs() < 1e-5, "got {}", out[1]);
    }

    #[test]
    fn hybrid_score_batch_length_mismatch_errors() {
        let r = hybrid_score_batch(
            &[1.0],
            &[1.0, 2.0],
            &[1],
            &[0.0],
            &[0.0],
            0.7,
            0.1,
            0.15,
            40,
        );
        assert!(r.is_err());
    }

    #[test]
    fn recency_bonus_ranks_linear_interpolation() {
        let v = vec![
            Some("2025-01-01".to_string()),
            Some("".to_string()),
            None,
            Some("2026-01-01".to_string()),
            Some("2025-06-01".to_string()),
        ];
        let b = recency_bonus_ranks(&v, 1.0);
        // dated: idx 0 (2025-01-01, rank 0), idx 4 (2025-06-01, rank 1), idx 3 (2026-01-01, rank 2)
        // denom = 2
        assert_eq!(b[1], 0.0);
        assert_eq!(b[2], 0.0);
        assert!((b[0] - 0.0).abs() < 1e-6);
        assert!((b[4] - 0.5).abs() < 1e-6);
        assert!((b[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_simd_matches_scalar_reference_various_lengths() {
        // Inline scalar reference: simplest possible accumulation, no SIMD.
        fn cosine_scalar_ref(a: &[f32], b: &[f32]) -> f32 {
            let mut d = 0.0f32;
            let mut sa = 0.0f32;
            let mut sb = 0.0f32;
            for i in 0..a.len() {
                d += a[i] * b[i];
                sa += a[i] * a[i];
                sb += b[i] * b[i];
            }
            if sa == 0.0 || sb == 0.0 {
                return 0.0;
            }
            (d / (sa.sqrt() * sb.sqrt())).clamp(-1.0, 1.0)
        }
        // Cheap deterministic xorshift PRNG to avoid extra deps in the test.
        let mut state: u32 = 0xC0FFEE;
        let mut rand_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Map u32 -> f32 in [-1, 1).
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for &len in &[1usize, 7, 8, 9, 16, 100, 1024] {
            let a: Vec<f32> = (0..len).map(|_| rand_f32()).collect();
            let b: Vec<f32> = (0..len).map(|_| rand_f32()).collect();
            let got = cosine_unchecked(&a, &b);
            let want = cosine_scalar_ref(&a, &b);
            assert!(
                (got - want).abs() < 1e-5,
                "len={len}: got={got}, want={want}, diff={}",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn recency_bonus_ranks_degenerate_cases() {
        assert_eq!(recency_bonus_ranks(&[], 1.0), Vec::<f32>::new());
        assert_eq!(
            recency_bonus_ranks(&[Some("x".to_string())], 1.0),
            vec![0.0]
        );
        assert_eq!(recency_bonus_ranks(&[None, None], 1.0), vec![0.0, 0.0]);
        // zero bias -> all zero regardless
        let v = vec![Some("a".to_string()), Some("b".to_string())];
        assert_eq!(recency_bonus_ranks(&v, 0.0), vec![0.0, 0.0]);
    }

    fn unit_vectors(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| vec![(i as f32).cos(), (i as f32).sin(), 0.5])
            .collect()
    }

    #[test]
    fn mmr_nan_relevance_ranks_last_on_both_branches() {
        // 200 candidates take the rayon branch, 50 the sequential one; NaN
        // panicked the former and silently won position 0 in the latter.
        for n in [50usize, 200] {
            let vs = unit_vectors(n);
            let refs: Vec<&[f32]> = vs.iter().map(|v| v.as_slice()).collect();
            let mut rel: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 / n as f32).collect();
            rel[0] = f32::NAN;
            let out = mmr_rerank_scored(&rel, &refs, 1.0, n, false).unwrap();
            assert_eq!(out.len(), n);
            assert_eq!(
                *out.last().unwrap(),
                0,
                "n={n}: NaN candidate must rank last, got {out:?}"
            );
        }
    }

    #[test]
    fn mmr_pick_sequential_and_parallel_agree_on_ties_and_nan() {
        for n in [MMR_PAR_THRESHOLD - 1, MMR_PAR_THRESHOLD + 1] {
            let remaining: Vec<usize> = (0..n).collect();
            // Two equal maxima at 7 and 90 (when present); NaN everywhere else.
            let best = mmr_pick(
                &remaining,
                |c| if c == 7 || c == 90 { 1.0 } else { f32::NAN },
            );
            assert_eq!(best, Some(7), "n={n}: a tie must go to the lower position");
            let all_nan = mmr_pick(&remaining, |_| f32::NAN);
            assert_eq!(
                all_nan,
                Some(0),
                "n={n}: all-NaN picks position 0 on both branches"
            );
        }
        assert_eq!(mmr_pick(&[], |_| 0.0), None);
    }

    #[test]
    fn misaligned_packed_rows_score_like_aligned_ones() {
        let q: &[f32] = &[1.0, 2.0, 3.0, 4.0];
        let row = f32_as_blob(q).to_vec();
        let mut buf = vec![0u8; 1];
        buf.extend_from_slice(&row); // offset by one byte: never 4-aligned
        let blobs: Vec<&[u8]> = vec![&buf[1..]];
        let got = cosine_batch_packed(q, &blobs, 4).unwrap();
        assert!(
            (got[0] - 1.0).abs() < 1e-6,
            "misaligned row scored {}",
            got[0]
        );
    }

    #[test]
    fn dim_mismatch_names_the_row() {
        let q: &[f32] = &[1.0, 0.0];
        let good: &[f32] = &[1.0, 0.0];
        let bad: &[f32] = &[1.0, 0.0, 0.0];
        let e = cosine_batch(q, &[good, bad]).unwrap_err();
        assert!(
            matches!(
                e,
                M3Error::VectorDimMismatch {
                    expected: 2,
                    got: 3,
                    index: Some(1)
                }
            ),
            "{e}"
        );
        assert!(e.to_string().contains("at row 1"), "{e}");
    }

    #[test]
    fn recency_parse_saturates_instead_of_overflowing() {
        let v = vec![
            Some("99999999999999999999999-01-01".to_string()),
            Some("2025-01-01".to_string()),
        ];
        let b = recency_bonus_ranks(&v, 1.0);
        assert_eq!(
            b,
            vec![1.0, 0.0],
            "an absurd year sorts newest, without a panic"
        );
    }

    #[test]
    fn token_jaccard_parity() {
        // "Alex Johnson," -> "alex johnson"
        // "Alex Johnson" -> "alex johnson"
        assert!((token_jaccard("Alex Johnson,", "Alex Johnson") - 1.0).abs() < 1e-6);
        // "Apple Inc." vs "Apple"
        // ta = {apple, inc}, tb = {apple} -> 1/2 = 0.5
        assert!((token_jaccard("Apple Inc.", "Apple") - 0.5).abs() < 1e-6);
        // No tokens
        assert_eq!(token_jaccard("!!!", "???"), 0.0);
    }
}
