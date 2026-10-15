//! Token-exact sliding windows over a text.
//!
//! m3 chunked long memories with a token ESTIMATE that may over-count but never
//! under-counts (`max(bytes/3, chars) + 2`). On English prose it over-counts about
//! 3x, so m3 cut ~1.7x more windows than a real count needs (measured
//! 2026-10-15 on 110 memories over 6000 chars: 192 windows vs 112), and every
//! extra window is one more embed call.
//!
//! [`plan_windows`] cuts windows by REAL token counts instead. It is generic over
//! the counter so the planning logic is tested without a model; the embed
//! backend supplies its tokenizer.
//!
//! Boundaries are CHARACTER offsets into the original text (Unicode scalar
//! values, which is what a Python `str` index is). Each window's end is found by
//! binary search on the count, so every window is proven to fit `budget` by an
//! actual count of exactly that slice, and the text is never re-assembled from
//! tokens (SentencePiece normalisation would change it).

use m3_error::Result;

/// Upper bound on characters per token used to cap a search range, so a very
/// long text does not tokenize its whole remainder for every window. A window
/// that would need more characters than this per token is simply cut at the cap.
pub const MAX_CHARS_PER_TOKEN: usize = 32;

/// Cut `[0, n_chars)` into windows of at most `budget` tokens each, overlapping by
/// about `overlap` tokens. `count(start, end)` returns the token count of the
/// character slice `[start, end)`, including any frame tokens the embed path adds.
///
/// Returns `[(start, end), ...]` covering the whole text, in order. A text that
/// fits returns exactly one window `(0, n_chars)`; an empty text returns
/// `[(0, 0)]`. A single character over budget becomes its own window rather
/// than stalling (the embed path then reports it).
pub fn plan_windows(
    n_chars: usize,
    budget: usize,
    overlap: usize,
    count: &mut dyn FnMut(usize, usize) -> Result<usize>,
) -> Result<Vec<(usize, usize)>> {
    if n_chars == 0 {
        return Ok(vec![(0, 0)]);
    }
    if count(0, n_chars)? <= budget {
        return Ok(vec![(0, n_chars)]);
    }
    let overlap = overlap.min(budget.saturating_sub(1));
    let mut out = Vec::new();
    let mut start = 0usize;
    loop {
        let end = largest_fitting_end(start, n_chars, budget, count)?;
        out.push((start, end));
        if end >= n_chars {
            return Ok(out);
        }
        let next = smallest_overlap_start(start, end, overlap, count)?;
        // Always advance: a pathological count must not loop forever.
        start = if next > start { next } else { end };
    }
}

/// The largest `end` in `(start, n]` whose slice fits `budget`; at least
/// `start + 1` so a lone oversized character still makes progress.
fn largest_fitting_end(
    start: usize,
    n: usize,
    budget: usize,
    count: &mut dyn FnMut(usize, usize) -> Result<usize>,
) -> Result<usize> {
    let cap = n.min(start.saturating_add(budget.saturating_mul(MAX_CHARS_PER_TOKEN)));
    if count(start, cap)? <= budget {
        return Ok(cap);
    }
    // Invariant: count(start, lo) <= budget (or lo == start), count(start, hi) > budget.
    let (mut lo, mut hi) = (start, cap);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if count(start, mid)? <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(if lo > start { lo } else { start + 1 })
}

/// The smallest `s` in `(start, end)` whose slice `[s, end)` holds at most
/// `overlap` tokens, so the next window repeats about `overlap` tokens.
fn smallest_overlap_start(
    start: usize,
    end: usize,
    overlap: usize,
    count: &mut dyn FnMut(usize, usize) -> Result<usize>,
) -> Result<usize> {
    if overlap == 0 {
        return Ok(end);
    }
    // Invariant: count(lo, end) > overlap (or lo == start), count(hi, end) <= overlap.
    let (mut lo, mut hi) = (start, end);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if count(mid, end)? <= overlap {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counter that charges `per` characters per token plus a 2-token frame,
    /// the shape of the real tokenizer (BOS + EOS) on uniform text.
    fn uniform(per: usize) -> impl FnMut(usize, usize) -> Result<usize> {
        move |s, e| Ok((e - s).div_ceil(per) + 2)
    }

    fn check_cover(w: &[(usize, usize)], n: usize) {
        assert_eq!(w.first().unwrap().0, 0, "starts at 0");
        assert_eq!(w.last().unwrap().1, n, "reaches the end");
        for pair in w.windows(2) {
            assert!(pair[1].0 > pair[0].0, "advances: {pair:?}");
            assert!(pair[1].0 <= pair[0].1, "no gap between windows: {pair:?}");
        }
    }

    #[test]
    fn a_text_that_fits_is_one_window() {
        let mut c = uniform(4);
        assert_eq!(plan_windows(100, 100, 10, &mut c).unwrap(), vec![(0, 100)]);
        assert_eq!(plan_windows(0, 100, 10, &mut c).unwrap(), vec![(0, 0)]);
    }

    #[test]
    fn every_window_fits_and_the_text_is_covered() {
        let mut c = uniform(4);
        let n = 100_000;
        let w = plan_windows(n, 7000, 2000, &mut c).unwrap();
        check_cover(&w, n);
        for &(s, e) in &w {
            assert!(c(s, e).unwrap() <= 7000, "window {s}..{e} over budget");
        }
    }

    #[test]
    fn windows_are_as_large_as_the_budget_allows() {
        // 4 chars/token, budget 7000 -> (7000 - 2) * 4 = 27_992 chars per window.
        let mut c = uniform(4);
        let w = plan_windows(100_000, 7000, 0, &mut c).unwrap();
        assert_eq!(w[0], (0, 27_992));
        assert_eq!(
            w.len(),
            4,
            "100k chars at ~28k per window, no overlap: {w:?}"
        );
    }

    #[test]
    fn the_overlap_is_about_the_requested_tokens() {
        let mut c = uniform(4);
        let w = plan_windows(100_000, 7000, 2000, &mut c).unwrap();
        let (a, b) = (w[0], w[1]);
        let shared = c(b.0, a.1).unwrap();
        assert!(shared <= 2000 && shared >= 1990, "overlap {shared} tokens");
    }

    #[test]
    fn exact_counts_need_fewer_windows_than_the_estimate() {
        // The estimator charges 1 token per char on prose that is really ~4 per token.
        let n = 239_453;
        let mut real = uniform(4);
        let mut estimate = uniform(1);
        let exact = plan_windows(n, 7000, 2000, &mut real).unwrap().len();
        let estimated = plan_windows(n, 7000, 2000, &mut estimate).unwrap().len();
        assert!(
            exact * 3 < estimated,
            "exact {exact} vs estimated {estimated}"
        );
    }

    #[test]
    fn an_oversized_character_still_makes_progress() {
        // Every character costs more than the whole budget.
        let mut c = |s: usize, e: usize| Ok((e - s) * 50 + 2);
        let w = plan_windows(5, 10, 3, &mut c).unwrap();
        assert_eq!(w, vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
    }

    #[test]
    fn a_counter_error_is_returned_not_swallowed() {
        let mut c = |_s: usize, _e: usize| Err(m3_error::M3Error::Backend("tokenizer gone".into()));
        assert!(plan_windows(10, 5, 1, &mut c).is_err());
    }
}
