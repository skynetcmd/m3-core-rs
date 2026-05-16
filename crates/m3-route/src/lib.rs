//! Multi-signal query route decider (Phase 3d §4c.5).
//!
//! Port of `bin/auto_route.py`. The riskiest §4c port: behavior-changing, not
//! byte-exact. Gated by shadow mode — branch identity must match Python for
//! >=99.9% of a frozen 10k-query corpus before cutover.
//!
//! NOTE: shadow-mode behavior-parity testing against the Python 10k-query
//! corpus is pending — the corpus is not yet available in this workspace. The
//! unit tests below cover branch selection, signal extraction, and tie-break
//! determinism as an interim guard.

use std::sync::OnceLock;
use regex::Regex;

/// Signals extracted from a query, fed into the scoring function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSignals {
    pub token_count: usize,
    /// A capitalized multi-word span or quoted span hinting at a named entity.
    pub has_entity_hint: bool,
    /// A recency cue like "yesterday" / "last week" / "recently".
    pub recency_cue: bool,
    /// An explicit intent verb, if found ("find", "compare", "summarize", ...).
    pub intent_marker: Option<String>,
    /// The query is phrased as a question.
    pub question_form: bool,
}

/// The decision: chosen branch plus confidence and per-signal breakdown.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub branch: String,
    pub confidence: f32,
    pub signal_breakdown: Vec<(String, f32)>,
}

/// Fixed branch set. Tie-break is lexicographic on these names.
const BRANCHES: [&str; 4] = ["entity", "lexical", "semantic", "temporal"];

// Indices into the scores array — must match BRANCHES order (alphabetical).
const BRANCH_ENTITY: usize = 0;
const BRANCH_LEXICAL: usize = 1;
const BRANCH_SEMANTIC: usize = 2;
const BRANCH_TEMPORAL: usize = 3;

fn recency_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(yesterday|today|tonight|recently|last\s+(?:week|month|year|night)|ago|this\s+(?:week|month|morning)|earlier)\b").unwrap()
    })
}

fn intent_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(find|search|compare|summarize|summarise|list|show|explain|define|recall|remember)\b").unwrap()
    })
}

fn entity_hint_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Quoted span, or two+ consecutive Capitalized words.
    RE.get_or_init(|| {
        Regex::new(r#"("[^"]+"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)"#).unwrap()
    })
}

/// Pure feature extraction from a raw query string.
pub fn extract_signals(query: &str) -> RouteSignals {
    let token_count = query.split_whitespace().filter(|t| !t.is_empty()).count();
    let recency_cue = recency_re().is_match(query);
    let has_entity_hint = entity_hint_re().is_match(query);
    let intent_marker = intent_re()
        .find(query)
        .map(|m| m.as_str().to_lowercase());
    let trimmed = query.trim_end();
    let question_form = trimmed.ends_with('?') || {
        // First-word ASCII case-insensitive check — allocation-free.
        let first_word = query.split_whitespace().next().unwrap_or("");
        const PREFIXES: &[&str] = &[
            "who", "what", "when", "where", "why", "how",
            "which", "did", "is", "are", "was", "were",
        ];
        PREFIXES.iter().any(|&p| first_word.eq_ignore_ascii_case(p))
    };

    RouteSignals { token_count, has_entity_hint, recency_cue, intent_marker, question_form }
}

/// Multi-signal weighted scoring. Returns the chosen branch, its confidence
/// (winning score normalized by total score), and the full per-branch
/// breakdown. Ties are broken lexicographically on branch name — deterministic
/// and explicitly tested.
pub fn decide_route(query: &str, signals: &RouteSignals) -> RouteDecision {
    let _ = query;
    let mut scores = [0.0f32; 4];

    // temporal: recency cues dominate.
    if signals.recency_cue {
        scores[BRANCH_TEMPORAL] += 3.0;
    }
    // entity: entity-shaped spans.
    if signals.has_entity_hint {
        scores[BRANCH_ENTITY] += 3.0;
    }
    // lexical: short keyword-style queries with no question form.
    if signals.token_count <= 3 {
        scores[BRANCH_LEXICAL] += 2.0;
    }
    if !signals.question_form {
        scores[BRANCH_LEXICAL] += 1.0;
    }
    // semantic: longer, natural-language / question-form queries.
    if signals.question_form {
        scores[BRANCH_SEMANTIC] += 2.0;
    }
    if signals.token_count >= 6 {
        scores[BRANCH_SEMANTIC] += 2.0;
    }
    // intent markers nudge toward semantic unless they are recall-flavoured.
    if let Some(intent) = &signals.intent_marker {
        match intent.as_str() {
            "recall" | "remember" => scores[BRANCH_TEMPORAL] += 1.0,
            "compare" | "explain" | "define" | "summarize" | "summarise" => {
                scores[BRANCH_SEMANTIC] += 1.5
            }
            _ => scores[BRANCH_SEMANTIC] += 0.5,
        }
    }
    // Baseline so an all-zero query still resolves deterministically.
    scores[BRANCH_SEMANTIC] += 0.1;

    // Pick max score; lexicographic tie-break — BRANCHES is alphabetical, so
    // iterate in index order and only replace on strictly-greater. The lowest
    // index (lexicographically first name) wins ties.
    let mut best_idx = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    let total: f32 = scores.iter().sum();
    let confidence = if total > 0.0 { best_score / total } else { 0.0 };

    let signal_breakdown = BRANCHES
        .iter()
        .zip(scores.iter())
        .map(|(b, s)| ((*b).to_string(), *s))
        .collect();

    RouteDecision {
        branch: BRANCHES[best_idx].to_string(),
        confidence,
        signal_breakdown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_recency_and_question() {
        let s = extract_signals("What did I do yesterday?");
        assert!(s.recency_cue);
        assert!(s.question_form);
        assert_eq!(s.token_count, 5);
    }

    #[test]
    fn extract_entity_hint() {
        let s = extract_signals("notes about Project Oxidation");
        assert!(s.has_entity_hint);
        let quoted = extract_signals(r#"the "rank fusion merge" task"#);
        assert!(quoted.has_entity_hint);
        let none = extract_signals("the quick brown fox");
        assert!(!none.has_entity_hint);
    }

    #[test]
    fn extract_intent_marker() {
        assert_eq!(extract_signals("compare these two").intent_marker.as_deref(), Some("compare"));
        assert_eq!(extract_signals("just chatting").intent_marker, None);
    }

    #[test]
    fn route_temporal() {
        let q = "what changed last week";
        let d = decide_route(q, &extract_signals(q));
        assert_eq!(d.branch, "temporal");
    }

    #[test]
    fn route_entity() {
        let q = "Project Oxidation status";
        let d = decide_route(q, &extract_signals(q));
        assert_eq!(d.branch, "entity");
    }

    #[test]
    fn route_lexical() {
        let q = "cosine simd";
        let d = decide_route(q, &extract_signals(q));
        assert_eq!(d.branch, "lexical");
    }

    #[test]
    fn route_semantic() {
        let q = "how does the hybrid rank fusion merge feed into mmr reranking";
        let d = decide_route(q, &extract_signals(q));
        assert_eq!(d.branch, "semantic");
    }

    #[test]
    fn tie_break_is_lexicographic_and_deterministic() {
        // Construct signals that produce an exact tie between branches.
        // token_count 4, not question, no other cues:
        //   lexical gets 1.0 (not question), semantic gets 0.1 baseline.
        // Force a tie by hand-building signals where entity and temporal both
        // score 3.0.
        let signals = RouteSignals {
            token_count: 10,
            has_entity_hint: true,
            recency_cue: true,
            intent_marker: None,
            question_form: false,
        };
        // entity=3.0, temporal=3.0, lexical=1.0, semantic=2.1 -> tie at 3.0.
        let d1 = decide_route("x", &signals);
        let d2 = decide_route("x", &signals);
        assert_eq!(d1.branch, d2.branch, "must be deterministic");
        assert_eq!(d1.branch, "entity", "lexicographically first of the tied pair");
    }

    #[test]
    fn confidence_in_unit_range() {
        let q = "what happened yesterday";
        let d = decide_route(q, &extract_signals(q));
        assert!((0.0..=1.0).contains(&d.confidence));
        assert_eq!(d.signal_breakdown.len(), 4);
    }

    #[test]
    fn empty_query_resolves_deterministically() {
        let d1 = decide_route("", &extract_signals(""));
        let d2 = decide_route("", &extract_signals(""));
        assert_eq!(d1.branch, d2.branch);
        // No cues, short, non-question -> lexical (short + non-question weight).
        assert_eq!(d1.branch, "lexical");
    }
}
