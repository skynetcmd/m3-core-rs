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

/// Reorder `items` in place to enforce the displacement policy.
///
/// No-op if `protected_ranks == 0` or `margin <= 1.0`. Idempotent on
/// already-conforming input.
pub fn enforce_displacement_guard(
    items: &mut [DisplacementRow],
    protected_ranks: usize,
    margin: f32,
) {
    let n = items.len();
    if n == 0 || protected_ranks == 0 || margin <= 1.0 {
        return;
    }
    let limit = protected_ranks.min(n);
    for rank in 0..limit {
        if !items[rank].is_expansion {
            continue;
        }
        let mut next_primary: Option<usize> = None;
        for j in (rank + 1)..n {
            if !items[j].is_expansion {
                next_primary = Some(j);
                break;
            }
        }
        let Some(pidx) = next_primary else {
            continue;
        };
        let score = items[rank].score;
        let primary_score = items[pidx].score;
        if score > 0.0 && primary_score > 0.0 && score >= margin * primary_score {
            continue;
        }
        items.swap(rank, pidx);
    }
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
