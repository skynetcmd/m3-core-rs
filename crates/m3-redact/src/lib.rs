//! Config-driven secret redaction.
//!
//! Byte-exact functional port of m3-memory's `bin/chatlog_redaction.py`.
//! Same built-in pattern groups, same per-pattern order, same evaluation
//! order, same `[REDACTED:<group>]` replacement marker, same return shape.
//!
//! Caching: a `Redactor` compiles its config once on construction. The Python
//! side caches compiled patterns by a blake2b hash of the config tuple; here
//! the caller owns the `Redactor` lifetime, so recompilation is just "build a
//! new Redactor". The convenience `scrub()` fn builds a throwaway Redactor per
//! call — faithful to Python's per-call `compile_patterns()` (which is a no-op
//! when the config hash is unchanged, so amortized cost is the same).

use regex::Regex;

/// Redaction configuration — mirrors the `redaction` sub-dict of the chat log
/// config consumed by `chatlog_redaction.py`.
#[derive(Debug, Clone)]
pub struct RedactionConfig {
    pub enabled: bool,
    /// Which built-in groups (and `"custom_regex"`, `"pii"`) are active.
    pub patterns: Vec<String>,
    pub custom_regex: Vec<String>,
    pub redact_pii: bool,
}

impl Default for RedactionConfig {
    fn default() -> Self {
        RedactionConfig {
            enabled: false,
            patterns: Vec::new(),
            custom_regex: Vec::new(),
            redact_pii: false,
        }
    }
}

/// Outcome of a `scrub` pass.
#[derive(Debug, Clone)]
pub struct ScrubResult {
    pub content: String,
    pub match_count: usize,
    /// Group names that had >=1 match, in evaluation order.
    pub groups_fired: Vec<String>,
    /// custom_regex compilation errors (bad patterns are skipped, not fatal).
    pub compile_errors: Vec<String>,
}

/// Evaluation order — must match chatlog_redaction.py line ~194.
const EVALUATION_ORDER: &[&str] = &[
    "api_keys",
    "bearer_tokens",
    "jwt",
    "aws_keys",
    "github_tokens",
    "custom_regex",
    "pii",
];

/// Built-in pattern groups, ported verbatim from chatlog_redaction.py
/// lines ~55-135. Each entry is `(pattern_name, regex_source)`; pattern_name
/// is unused in output (the GROUP name is emitted) but kept for fidelity.
fn builtin_group(name: &str) -> Option<Vec<(&'static str, &'static str)>> {
    Some(match name {
        "api_keys" => vec![
            ("anthropic", r"sk-ant-[A-Za-z0-9_-]{20,}"),
            ("openai_project", r"sk-proj-[A-Za-z0-9_-]{20,}"),
            ("openai_generic", r"sk-[A-Za-z0-9]{20,}"),
            ("xai", r"xai-[A-Za-z0-9_-]{20,}"),
            ("google", r"AIza[0-9A-Za-z_-]{35}"),
        ],
        "bearer_tokens" => vec![
            (
                "auth_header",
                r"(?i)Authorization:\s*Bearer\s+[A-Za-z0-9._~+/=-]{20,}",
            ),
            ("bearer_generic", r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
        ],
        "jwt" => vec![(
            "jwt",
            r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        )],
        "aws_keys" => vec![
            ("access_key_id", r"AKIA[0-9A-Z]{16}"),
            (
                "secret_access_key",
                r#"(?i)aws[_-]?secret[_-]?access[_-]?key["'\s:=]+[A-Za-z0-9/+=]{40}"#,
            ),
        ],
        "github_tokens" => vec![
            ("ghp", r"ghp_[A-Za-z0-9]{36}"),
            ("gho", r"gho_[A-Za-z0-9]{36}"),
            ("ghu", r"ghu_[A-Za-z0-9]{36}"),
            ("ghs", r"ghs_[A-Za-z0-9]{36}"),
            ("ghr", r"ghr_[A-Za-z0-9]{36}"),
        ],
        "pii" => vec![
            ("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            (
                "us_phone",
                r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            ),
            ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
        ],
        _ => return None,
    })
}

/// A compiled, ready-to-apply redactor for one `RedactionConfig`.
pub struct Redactor {
    enabled: bool,
    /// (group_name, [compiled patterns]) in evaluation order, only for active groups.
    groups: Vec<(String, Vec<Regex>)>,
    compile_errors: Vec<String>,
}

impl Redactor {
    /// Compile a config. Mirrors `compile_patterns()`. Never fails: bad
    /// custom_regex entries are collected into `compile_errors` and skipped.
    pub fn new(config: &RedactionConfig) -> Self {
        let mut groups: Vec<(String, Vec<Regex>)> = Vec::new();
        let mut compile_errors: Vec<String> = Vec::new();

        if !config.enabled {
            return Redactor {
                enabled: false,
                groups,
                compile_errors,
            };
        }

        let pat_set: Vec<&str> = config.patterns.iter().map(|s| s.as_str()).collect();

        for &group_name in EVALUATION_ORDER {
            match group_name {
                "pii" => {
                    // PII only active if "pii" in patterns AND redact_pii.
                    if !(pat_set.contains(&"pii") && config.redact_pii) {
                        continue;
                    }
                    if let Some(specs) = builtin_group("pii") {
                        let compiled: Vec<Regex> = specs
                            .iter()
                            .map(|(_, src)| Regex::new(src).expect("builtin pii regex must compile"))
                            .collect();
                        groups.push(("pii".to_string(), compiled));
                    }
                }
                "custom_regex" => {
                    if !(pat_set.contains(&"custom_regex") && !config.custom_regex.is_empty()) {
                        continue;
                    }
                    let mut compiled: Vec<Regex> = Vec::new();
                    for pat in &config.custom_regex {
                        match Regex::new(pat) {
                            Ok(re) => compiled.push(re),
                            Err(e) => compile_errors.push(format!(
                                "custom_regex compilation error: {pat}: {e}"
                            )),
                        }
                    }
                    if !compiled.is_empty() {
                        groups.push(("custom_regex".to_string(), compiled));
                    }
                }
                _ => {
                    if !pat_set.contains(&group_name) {
                        continue;
                    }
                    if let Some(specs) = builtin_group(group_name) {
                        let compiled: Vec<Regex> = specs
                            .iter()
                            .map(|(_, src)| {
                                Regex::new(src).expect("builtin regex must compile")
                            })
                            .collect();
                        groups.push((group_name.to_string(), compiled));
                    }
                }
            }
        }

        Redactor {
            enabled: true,
            groups,
            compile_errors,
        }
    }

    /// custom_regex compile errors from construction (parity with
    /// Python `get_compile_errors()`).
    pub fn compile_errors(&self) -> &[String] {
        &self.compile_errors
    }

    /// Apply this redactor to `content`. Mirrors the body of `scrub()`.
    pub fn apply(&self, content: &str) -> ScrubResult {
        if !self.enabled {
            return ScrubResult {
                content: content.to_string(),
                match_count: 0,
                groups_fired: Vec::new(),
                compile_errors: self.compile_errors.clone(),
            };
        }

        let mut scrubbed = content.to_string();
        let mut total_matches = 0usize;
        let mut groups_fired: Vec<String> = Vec::new();

        for (group_name, patterns) in &self.groups {
            let mut group_matches = 0usize;
            let marker = format!("[REDACTED:{group_name}]");
            for re in patterns {
                // Count matches against the CURRENT string (earlier patterns
                // in the group already mutated it), then replace all.
                group_matches += re.find_iter(&scrubbed).count();
                scrubbed = re.replace_all(&scrubbed, marker.as_str()).into_owned();
            }
            if group_matches > 0 {
                total_matches += group_matches;
                groups_fired.push(group_name.clone());
            }
        }

        ScrubResult {
            content: scrubbed,
            match_count: total_matches,
            groups_fired,
            compile_errors: self.compile_errors.clone(),
        }
    }
}

/// Convenience: build a throwaway `Redactor` and scrub in one call.
/// Faithful to Python's per-call `compile_patterns()` + `scrub()`.
///
/// If `config.enabled` is false, returns `(content, 0, [])` immediately.
pub fn scrub(content: &str, config: &RedactionConfig) -> ScrubResult {
    if !config.enabled {
        return ScrubResult {
            content: content.to_string(),
            match_count: 0,
            groups_fired: Vec::new(),
            compile_errors: Vec::new(),
        };
    }
    Redactor::new(config).apply(content)
}
