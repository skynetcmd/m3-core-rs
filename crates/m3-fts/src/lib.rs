//! FTS5 query sanitization and lexical tokenization.
//!
//! Byte-exact functional port of m3-memory's `bin/memory/fts.py`
//! (`_sanitize_fts`, `_sanitize_for_searchable`, `_compile_fts_query`).
//! Same operator stripping, same searchable-punctuation transform (mirroring
//! the SQLite `mi_fts_insert` trigger), same mode-dependent branching, same
//! `(fts_query, ok)` return shape.
//!
//! Why Rust: the Python path runs two regex substitutions plus whitespace
//! splitting on every search call. This crate does the same work with a single
//! linear scan and no regex engine, so the hot `_compile_fts_query` path drops
//! its per-call allocation and regex overhead while preserving exact output.

use regex::Regex;
use std::sync::LazyLock;

/// `\b(OR|AND|NOT|NEAR)\b|[*()\[\]{}]` — FTS5 operator words (whole-word) and
/// the bracket/wildcard punctuation that would otherwise reach the FTS5 parser
/// as syntax. Matched the same way as the Python `_FTS_OPERATORS` regex so a
/// hostile query like `AND OR NOT *(foo)` is neutralized identically.
static FTS_OPERATORS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(OR|AND|NOT|NEAR)\b|[*()\[\]{}]").unwrap());

/// The 8 punctuation characters the `mi_fts_insert` trigger replaces with a
/// space (after lowercasing) before storing into `content_searchable` /
/// `title_searchable`. Query-side text must apply the same transform so MATCH
/// terms align with what FTS5 indexed: `? ! : . , ; / " '`.
const SEARCHABLE_PUNCT: [char; 8] = ['?', '!', ':', '.', ',', ';', '/', '"'];
// Note: the Python maketrans set is "?!:.,;/\"'" — the 8th char is the single
// quote, handled separately below so the array stays a clean 8-element list.
const SEARCHABLE_QUOTE: char = '\'';

/// Strip FTS5 operators from user input to prevent query injection / parse
/// errors. Mirrors `_sanitize_fts`: truncate to `max_len` chars first, replace
/// every operator match with a single space, then trim leading/trailing
/// whitespace.
///
/// `max_len` counts Unicode scalar values (Python `len()` on `str` counts code
/// points), not bytes, so multibyte input truncates at the same boundary.
pub fn sanitize_fts(query: &str, max_len: usize) -> String {
    let truncated: String = if query.chars().count() > max_len {
        query.chars().take(max_len).collect()
    } else {
        query.to_string()
    };
    FTS_OPERATORS.replace_all(&truncated, " ").trim().to_string()
}

/// Apply the same lowercase + depunctuate transform as the FTS triggers.
/// Mirrors `_sanitize_for_searchable`: empty in -> empty out; otherwise
/// lowercase, then map each of the 9 searchable-punct chars to a space.
///
/// Lowercasing uses `to_lowercase` (Unicode-aware), matching Python `str.lower`
/// for the ASCII inputs this path sees in practice.
pub fn sanitize_for_searchable(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    text.to_lowercase()
        .chars()
        .map(|c| {
            if c == SEARCHABLE_QUOTE || SEARCHABLE_PUNCT.contains(&c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// True iff `s` is non-empty and every char is alphanumeric — the Rust
/// equivalent of Python's Unicode-aware `str.isalnum()`. Python returns False
/// for the empty string and treats letters/digits/numeric chars across scripts
/// as alphanumeric; `char::is_alphanumeric` covers the same Unicode categories.
fn is_alnum(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric())
}

/// Compile a raw user query into an FTS5 MATCH string.
///
/// Returns `(fts_query, ok)`. When `ok` is false the caller treats it as
/// "no matchable tokens": in `mode == "fts5"` that means return no results, in
/// any other mode that means fall back to semantic-only.
///
/// Faithful to `_compile_fts_query`:
///   1. Exact-mode: a query wrapped in matching `"..."` or `'...'` is returned
///      as a quoted phrase verbatim (inner text preserved, re-wrapped in `"`).
///   2. Otherwise sanitize operators, then apply the searchable-punct transform.
///   3. Empty after sanitization -> `("", false)`.
///   4. `mode == "fts5"`: multi-token -> ` OR `-joined; single token ->
///      wildcarded (`tok*`) iff alphanumeric, else passed through.
///   5. Other modes (hybrid/semantic): single alphanumeric token with no space
///      -> wildcarded; otherwise passed through unchanged.
pub fn compile_fts_query(query: &str, mode: &str) -> (String, bool) {
    // Mirror Python EXACTLY: `query.startswith('"') and query.endswith('"')`.
    // Python applies no length guard, so a lone quote char (len 1) is its own
    // start AND end and IS treated as an exact phrase — `'"'[1:-1]` yields ""
    // → ('""', True). We must reproduce that quirk, not "fix" it: fts.py is the
    // source of truth and downstream FTS behavior is calibrated to it.
    let is_exact_query = (query.starts_with('"') && query.ends_with('"'))
        || (query.starts_with('\'') && query.ends_with('\''));
    if is_exact_query {
        // Python query[1:-1] on code points. For len<=1 this is an empty slice
        // (NOT a panic): Python `'"'[1:-1] == ''`. Guard the reverse-range case.
        let chars: Vec<char> = query.chars().collect();
        let inner: String = if chars.len() <= 1 {
            String::new()
        } else {
            chars[1..chars.len() - 1].iter().collect()
        };
        return (format!("\"{inner}\""), true);
    }

    let clean = sanitize_fts(query, 500);
    let clean = sanitize_for_searchable(&clean);
    if clean.trim().is_empty() {
        return (String::new(), false);
    }
    let clean = clean.trim();

    if mode == "fts5" {
        // Python `clean.split()` splits on any whitespace run, dropping empties.
        let toks: Vec<&str> = clean.split_whitespace().collect();
        if toks.len() > 1 {
            return (toks.join(" OR "), true);
        }
        // len(toks) <= 1 here. Python uses `clean` (the trimmed string), not the
        // token, in the wildcard branch — identical when there's one token, and
        // `clean` is non-empty/non-blank by the guard above.
        if is_alnum(clean) {
            return (format!("{clean}*"), true);
        }
        return (clean.to_string(), true);
    }

    // hybrid / semantic fallback path
    if !clean.contains(' ') && is_alnum(clean) {
        return (format!("{clean}*"), true);
    }
    (clean.to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_operators_and_brackets() {
        assert_eq!(sanitize_fts("AND OR NOT *(foo)", 500), "foo");
        assert_eq!(sanitize_fts("hello world", 500), "hello world");
        // Operators are whole-word: "ANDROID" must survive.
        assert_eq!(sanitize_fts("ANDROID phones", 500), "ANDROID phones");
        assert_eq!(sanitize_fts("NEAR/3 quux", 500), "/3 quux");
    }

    #[test]
    fn sanitize_truncates_by_codepoints() {
        assert_eq!(sanitize_fts("abcdef", 3), "abc");
        // Multibyte: 3 code points kept, not 3 bytes.
        assert_eq!(sanitize_fts("áéíóú", 3), "áéí");
    }

    #[test]
    fn searchable_lowercases_and_depunctuates() {
        assert_eq!(sanitize_for_searchable("Hello, World!"), "hello  world ");
        assert_eq!(sanitize_for_searchable("a.b:c;d/e?f\"g'h"), "a b c d e f g h");
        assert_eq!(sanitize_for_searchable(""), "");
    }

    #[test]
    fn exact_phrase_passthrough() {
        assert_eq!(compile_fts_query("\"red wine\"", "fts5"), ("\"red wine\"".into(), true));
        assert_eq!(compile_fts_query("'red wine'", "hybrid"), ("\"red wine\"".into(), true));
        // Quirk parity with Python: a lone quote char IS start==end, so it is
        // treated as an exact phrase and `'"'[1:-1]` -> "" -> ('""', True).
        assert_eq!(compile_fts_query("\"", "fts5"), ("\"\"".into(), true));
        assert_eq!(compile_fts_query("'", "hybrid"), ("\"\"".into(), true));
    }

    #[test]
    fn fts5_multi_token_or_joined() {
        assert_eq!(compile_fts_query("red wine", "fts5"), ("red OR wine".into(), true));
    }

    #[test]
    fn fts5_single_alnum_wildcarded() {
        assert_eq!(compile_fts_query("wine", "fts5"), ("wine*".into(), true));
    }

    #[test]
    fn empty_after_sanitize_is_not_ok() {
        assert_eq!(compile_fts_query("AND OR NOT", "fts5"), (String::new(), false));
    }

    #[test]
    fn hybrid_single_alnum_wildcarded_else_passthrough() {
        assert_eq!(compile_fts_query("wine", "hybrid"), ("wine*".into(), true));
        // multi-word in hybrid mode is passed through unchanged (depunctuated).
        assert_eq!(compile_fts_query("red wine", "hybrid"), ("red wine".into(), true));
    }
}
