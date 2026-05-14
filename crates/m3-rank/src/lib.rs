//! Hybrid FTS5 + vector rank-fusion merge (Phase 3d §4c.4).
//!
//! Score normalization, weighted blend, and dedup in a single pass; hands the
//! deduped candidate set directly to the m3-vector MMR reranker. Parity gate:
//! identical top-K ordering on a frozen query set.

use m3_error::Result;
use m3_vector::mmr_rerank;
use std::collections::HashMap;

/// Which result set a `RankRow` came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankSource {
    Fts,
    Vector,
}

/// One ranked candidate from either the FTS5 or vector result set.
#[derive(Debug, Clone)]
pub struct RankRow {
    pub id: String,
    pub score: f32,
    pub source: RankSource,
}

/// Blend weights for the two result sets.
#[derive(Debug, Clone, Copy)]
pub struct FusionWeights {
    pub fts: f32,
    pub vector: f32,
}

impl Default for FusionWeights {
    fn default() -> Self {
        FusionWeights { fts: 0.5, vector: 0.5 }
    }
}

/// Min-max normalize scores into [0,1]. A degenerate range (all equal, or
/// single element) maps every score to 1.0 so the list still contributes.
fn min_max_normalize(rows: &mut [RankRow]) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for r in rows.iter() {
        lo = lo.min(r.score);
        hi = hi.max(r.score);
    }
    let range = hi - lo;
    for r in rows.iter_mut() {
        r.score = if range > 0.0 { (r.score - lo) / range } else { 1.0 };
    }
}

/// Fuse FTS5 and vector result sets: normalize each independently, apply the
/// weighted blend, dedup by id, return sorted descending by fused score.
///
/// Dedup policy: when an id appears in both sets we take the **max** of the two
/// weighted contributions rather than the sum. Max keeps fused scores within
/// the weighted [0,1] envelope (a sum would let dual-source ids dominate purely
/// for being in both lists, which is a recall artifact, not relevance).
pub fn fuse(
    mut fts: Vec<RankRow>,
    mut vector: Vec<RankRow>,
    weights: FusionWeights,
) -> Vec<RankRow> {
    min_max_normalize(&mut fts);
    min_max_normalize(&mut vector);

    let mut acc: HashMap<String, RankRow> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    let absorb = |row: RankRow, w: f32, acc: &mut HashMap<String, RankRow>, order: &mut Vec<String>| {
        let contribution = row.score * w;
        match acc.get_mut(&row.id) {
            Some(existing) => {
                if contribution > existing.score {
                    existing.score = contribution;
                }
            }
            None => {
                order.push(row.id.clone());
                acc.insert(
                    row.id.clone(),
                    RankRow { id: row.id, score: contribution, source: row.source },
                );
            }
        }
    };

    for r in fts {
        absorb(r, weights.fts, &mut acc, &mut order);
    }
    for r in vector {
        absorb(r, weights.vector, &mut acc, &mut order);
    }

    let mut fused: Vec<RankRow> = order.into_iter().map(|id| acc.remove(&id).unwrap()).collect();
    // Descending by score; ties broken by id for determinism.
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    fused
}

/// Convenience: fuse, then hand the fused candidate ordering to the m3-vector
/// MMR reranker. `candidate_vectors[i]` must correspond to `fused[i]` after the
/// fuse step — callers supply vectors already aligned to the fused order, so
/// this helper fuses first and expects the caller to have ordered vectors to
/// match. Returns the MMR-selected indices into the fused list.
pub fn fuse_then_mmr(
    fts: Vec<RankRow>,
    vector: Vec<RankRow>,
    weights: FusionWeights,
    query_vec: &[f32],
    candidate_vectors: &[&[f32]],
    lambda: f32,
    k: usize,
) -> Result<(Vec<RankRow>, Vec<usize>)> {
    let fused = fuse(fts, vector, weights);
    let selected = mmr_rerank(query_vec, candidate_vectors, lambda, k)?;
    Ok((fused, selected))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, score: f32, source: RankSource) -> RankRow {
        RankRow { id: id.to_string(), score, source }
    }

    #[test]
    fn normalization_bounds() {
        let mut rows = vec![
            row("a", 10.0, RankSource::Fts),
            row("b", 20.0, RankSource::Fts),
            row("c", 30.0, RankSource::Fts),
        ];
        min_max_normalize(&mut rows);
        assert_eq!(rows[0].score, 0.0);
        assert_eq!(rows[2].score, 1.0);
        assert!(rows.iter().all(|r| (0.0..=1.0).contains(&r.score)));
    }

    #[test]
    fn normalization_degenerate_range() {
        let mut rows = vec![row("a", 5.0, RankSource::Fts), row("b", 5.0, RankSource::Fts)];
        min_max_normalize(&mut rows);
        assert!(rows.iter().all(|r| r.score == 1.0));
    }

    #[test]
    fn dedup_takes_max_contribution() {
        let fts = vec![row("shared", 100.0, RankSource::Fts), row("f", 0.0, RankSource::Fts)];
        let vec_rows = vec![row("shared", 100.0, RankSource::Vector), row("v", 0.0, RankSource::Vector)];
        let fused = fuse(fts, vec_rows, FusionWeights { fts: 0.3, vector: 0.7 });
        // 3 unique ids.
        assert_eq!(fused.len(), 3);
        let shared = fused.iter().find(|r| r.id == "shared").unwrap();
        // max(1.0*0.3, 1.0*0.7) = 0.7, not the sum 1.0.
        assert!((shared.score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn ordering_descending_with_id_tiebreak() {
        let fts = vec![row("b", 1.0, RankSource::Fts), row("a", 1.0, RankSource::Fts)];
        let fused = fuse(fts, vec![], FusionWeights::default());
        // Equal scores -> id ascending.
        assert_eq!(fused[0].id, "a");
        assert_eq!(fused[1].id, "b");
    }

    #[test]
    fn weight_extreme_all_fts() {
        let fts = vec![row("f1", 1.0, RankSource::Fts), row("f2", 0.0, RankSource::Fts)];
        let vec_rows = vec![row("v1", 1.0, RankSource::Vector)];
        let fused = fuse(fts, vec_rows, FusionWeights { fts: 1.0, vector: 0.0 });
        let v1 = fused.iter().find(|r| r.id == "v1").unwrap();
        assert_eq!(v1.score, 0.0);
        assert_eq!(fused[0].id, "f1");
    }

    #[test]
    fn weight_extreme_all_vector() {
        let fts = vec![row("f1", 1.0, RankSource::Fts)];
        let vec_rows = vec![row("v1", 1.0, RankSource::Vector), row("v2", 0.0, RankSource::Vector)];
        let fused = fuse(fts, vec_rows, FusionWeights { fts: 0.0, vector: 1.0 });
        let f1 = fused.iter().find(|r| r.id == "f1").unwrap();
        assert_eq!(f1.score, 0.0);
        assert_eq!(fused[0].id, "v1");
    }

    #[test]
    fn empty_inputs() {
        assert!(fuse(vec![], vec![], FusionWeights::default()).is_empty());
        let only_fts = fuse(vec![row("a", 1.0, RankSource::Fts)], vec![], FusionWeights::default());
        assert_eq!(only_fts.len(), 1);
    }
}
