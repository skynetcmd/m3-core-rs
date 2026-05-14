//! Expansion-displacement guard.
//!
//! Port of m3-memory's `_enforce_expansion_displacement_guard`. At ranks
//! `0..protected_ranks`, an expansion row may only outrank the next primary
//! row below it if `expansion_score >= margin * primary_score` (both scores
//! strictly positive). Otherwise the primary is swapped up. The crate stays
//! generic: the caller classifies each row as expansion or primary.

/// One ranked row: its score and whether it is an expansion (vs. primary).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplacementRow {
    pub score: f32,
    pub is_expansion: bool,
}

/// Compute the index permutation that enforces the displacement policy.
///
/// Returns `perm` such that `perm[new_rank]` is the original index now at that
/// rank. Callers holding their own row objects can apply `perm` to reorder
/// them. No-op permutation (identity) if `protected_ranks == 0` or
/// `margin <= 1.0`. This is the single source of truth for the guard's walk;
/// `enforce_displacement_guard` is a thin wrapper that applies the result.
pub fn displacement_permutation(
    items: &[DisplacementRow],
    protected_ranks: usize,
    margin: f32,
) -> Vec<usize> {
    let n = items.len();
    let mut perm: Vec<usize> = (0..n).collect();
    if n == 0 || protected_ranks == 0 || margin <= 1.0 {
        return perm;
    }
    // Work on a local copy so the swap-and-find-next-primary walk sees the
    // same evolving order the in-place variant would.
    let mut work: Vec<DisplacementRow> = items.to_vec();
    let limit = protected_ranks.min(n);
    for rank in 0..limit {
        if !work[rank].is_expansion {
            continue;
        }
        let mut next_primary: Option<usize> = None;
        for j in (rank + 1)..n {
            if !work[j].is_expansion {
                next_primary = Some(j);
                break;
            }
        }
        let Some(pidx) = next_primary else {
            continue;
        };
        let score = work[rank].score;
        let primary_score = work[pidx].score;
        if score > 0.0 && primary_score > 0.0 && score >= margin * primary_score {
            continue;
        }
        work.swap(rank, pidx);
        perm.swap(rank, pidx);
    }
    perm
}

/// Reorder `items` in place to enforce the displacement policy.
///
/// No-op if `protected_ranks == 0` or `margin <= 1.0`. Idempotent on
/// already-conforming input.
pub fn enforce_displacement_guard(
    items: &mut [DisplacementRow],
    protected_ranks: usize,
    margin: f32,
) {
    let perm = displacement_permutation(items, protected_ranks, margin);
    let reordered: Vec<DisplacementRow> = perm.iter().map(|&i| items[i]).collect();
    items.copy_from_slice(&reordered);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(score: f32, is_expansion: bool) -> DisplacementRow {
        DisplacementRow { score, is_expansion }
    }

    #[test]
    fn expansion_failing_margin_swaps() {
        let mut v = vec![row(1.0, true), row(0.9, false)];
        enforce_displacement_guard(&mut v, 3, 2.0);
        assert_eq!(v, vec![row(0.9, false), row(1.0, true)]);
    }

    #[test]
    fn expansion_passing_margin_stays() {
        let mut v = vec![row(3.0, true), row(1.0, false)];
        enforce_displacement_guard(&mut v, 3, 2.0);
        assert_eq!(v, vec![row(3.0, true), row(1.0, false)]);
    }

    #[test]
    fn nonpositive_primary_wins() {
        let mut v = vec![row(-1.0, true), row(-2.0, false)];
        enforce_displacement_guard(&mut v, 3, 2.0);
        assert_eq!(v, vec![row(-2.0, false), row(-1.0, true)]);
    }

    #[test]
    fn margin_le_one_noop() {
        let mut v = vec![row(1.0, true), row(0.9, false)];
        enforce_displacement_guard(&mut v, 3, 1.0);
        assert_eq!(v, vec![row(1.0, true), row(0.9, false)]);
    }

    #[test]
    fn protected_ranks_zero_noop() {
        let mut v = vec![row(1.0, true), row(0.9, false)];
        enforce_displacement_guard(&mut v, 0, 2.0);
        assert_eq!(v, vec![row(1.0, true), row(0.9, false)]);
    }

    #[test]
    fn idempotent() {
        let mut v = vec![row(1.0, true), row(0.9, false), row(0.5, true)];
        enforce_displacement_guard(&mut v, 3, 2.0);
        let once = v.clone();
        enforce_displacement_guard(&mut v, 3, 2.0);
        assert_eq!(v, once);
    }

    #[test]
    fn no_primary_below_leaves_expansion() {
        let mut v = vec![row(1.0, true), row(0.5, true)];
        enforce_displacement_guard(&mut v, 3, 2.0);
        assert_eq!(v, vec![row(1.0, true), row(0.5, true)]);
    }
}
