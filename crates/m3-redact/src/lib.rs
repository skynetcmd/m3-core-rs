//! Multi-pattern secret redaction (Phase 3d §4c.1).
//!
//! Port of `bin/chatlog_redaction.py`. Parity gate: byte-exact output match
//! against the existing chatlog redaction corpus.
//!
//! NOTE: The real Python-parity corpus test is pending — the Python regression
//! corpus is not yet available in this workspace. Until then, parity is
//! approximated by the per-pattern unit tests in `tests/`.

use aho_corasick::AhoCorasick;
use regex::Regex;
use std::collections::BTreeMap;

/// A single named regex pattern plus the kind label used in its marker.
struct NamedPattern {
    kind: &'static str,
    re: Regex,
}

/// A named redaction profile (pattern set + fixed-prefix scanner).
pub struct RedactionProfile {
    pub name: String,
    patterns: Vec<NamedPattern>,
    /// Fixed known prefixes scanned with Aho-Corasick; each maps to a kind.
    prefix_ac: AhoCorasick,
    prefix_kinds: Vec<&'static str>,
}

/// The outcome of redacting one text blob.
pub struct RedactionResult {
    pub text: String,
    pub hits: usize,
    /// Count of redactions per kind label.
    pub breakdown: BTreeMap<String, usize>,
}

impl Default for RedactionProfile {
    fn default() -> Self {
        // Order matters: more specific patterns run before generic ones so a
        // token is labelled with its narrowest kind.
        let patterns = vec![
            NamedPattern {
                kind: "aws_access_key",
                re: Regex::new(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b").unwrap(),
            },
            NamedPattern {
                kind: "anthropic_key",
                re: Regex::new(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b").unwrap(),
            },
            NamedPattern {
                kind: "openai_key",
                re: Regex::new(r"\bsk-[A-Za-z0-9]{20,}\b").unwrap(),
            },
            NamedPattern {
                kind: "google_api_key",
                re: Regex::new(r"\bAIza[A-Za-z0-9_\-]{30,}\b").unwrap(),
            },
            NamedPattern {
                kind: "github_token",
                re: Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b").unwrap(),
            },
            NamedPattern {
                kind: "bearer_token",
                re: Regex::new(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b").unwrap(),
            },
            NamedPattern {
                kind: "email",
                re: Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap(),
            },
            NamedPattern {
                kind: "ipv4",
                re: Regex::new(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b").unwrap(),
            },
        ];

        let prefix_kinds = vec!["known_prefix", "known_prefix", "known_prefix"];
        let prefix_ac = AhoCorasick::new(["xoxb-", "xoxp-", "glpat-"]).unwrap();

        RedactionProfile {
            name: "default".to_string(),
            patterns,
            prefix_ac,
            prefix_kinds,
        }
    }
}

impl RedactionProfile {
    /// Construct a named profile reusing the default pattern set.
    pub fn named(name: impl Into<String>) -> Self {
        RedactionProfile { name: name.into(), ..RedactionProfile::default() }
    }
}

/// Extent of one fixed-prefix token: from the prefix start to the first
/// non-token character (token chars = alphanumeric, `-`, `_`).
fn prefix_token_end(text: &str, prefix_start: usize) -> usize {
    let bytes = text.as_bytes();
    let mut i = prefix_start;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphanumeric() || c == b'-' || c == b'_' {
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// Scrub secrets from `text` according to `profile`.
pub fn redact(text: &str, profile: &RedactionProfile) -> RedactionResult {
    // Collect non-overlapping match spans across all sources, then rebuild.
    // (start, end, kind). First-claimed span wins on overlap.
    let mut spans: Vec<(usize, usize, &'static str)> = Vec::new();

    let claim = |start: usize, end: usize, kind: &'static str, spans: &mut Vec<(usize, usize, &'static str)>| {
        if end <= start {
            return;
        }
        let overlaps = spans.iter().any(|&(s, e, _)| start < e && s < end);
        if !overlaps {
            spans.push((start, end, kind));
        }
    };

    for p in &profile.patterns {
        for m in p.re.find_iter(text) {
            claim(m.start(), m.end(), p.kind, &mut spans);
        }
    }
    for m in profile.prefix_ac.find_iter(text) {
        let kind = profile.prefix_kinds[m.pattern().as_usize()];
        let end = prefix_token_end(text, m.start());
        claim(m.start(), end, kind, &mut spans);
    }

    spans.sort_by_key(|&(s, _, _)| s);

    let mut out = String::with_capacity(text.len());
    let mut breakdown: BTreeMap<String, usize> = BTreeMap::new();
    let mut cursor = 0usize;
    for (start, end, kind) in &spans {
        if *start < cursor {
            continue;
        }
        out.push_str(&text[cursor..*start]);
        out.push_str(&format!("[REDACTED:{kind}]"));
        *breakdown.entry(kind.to_string()).or_insert(0) += 1;
        cursor = *end;
    }
    out.push_str(&text[cursor..]);

    let hits = breakdown.values().sum();
    RedactionResult { text: out, hits, breakdown }
}
