//! Parity tests mirroring chatlog_redaction.py's `__main__` self-tests
//! (lines ~230-281) plus per-group, eval-order, PII-gating, custom-regex,
//! bad-regex, and disabled-config coverage.

use m3_redact::{scrub, RedactionConfig, Redactor};

fn cfg(enabled: bool, patterns: &[&str], custom: &[&str], redact_pii: bool) -> RedactionConfig {
    RedactionConfig {
        enabled,
        patterns: patterns.iter().map(|s| s.to_string()).collect(),
        custom_regex: custom.iter().map(|s| s.to_string()).collect(),
        redact_pii,
    }
}

#[test]
fn disabled_config_is_noop() {
    let c = cfg(false, &[], &[], false);
    let r = scrub("sk-ant-foobar12345678901234567890", &c);
    assert_eq!(r.content, "sk-ant-foobar12345678901234567890");
    assert_eq!(r.match_count, 0);
    assert!(r.groups_fired.is_empty());
}

#[test]
fn api_keys_group() {
    let c = cfg(true, &["api_keys"], &[], false);
    let r = scrub(
        "here is sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 keep it safe",
        &c,
    );
    assert_eq!(r.match_count, 1);
    assert!(r.groups_fired.contains(&"api_keys".to_string()));
    assert!(r.content.contains("[REDACTED:api_keys]"));
}

#[test]
fn github_tokens_group() {
    let c = cfg(true, &["github_tokens"], &[], false);
    let input = format!("token: ghp_{}", "a".repeat(36));
    let r = scrub(&input, &c);
    assert_eq!(r.match_count, 1);
    assert!(r.content.contains("[REDACTED:github_tokens]"));
}

#[test]
fn bearer_tokens_group() {
    let c = cfg(true, &["bearer_tokens"], &[], false);
    let r = scrub("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123", &c);
    assert!(r.content.contains("[REDACTED:bearer_tokens]"));
    assert!(r.match_count >= 1);
}

#[test]
fn jwt_group() {
    let c = cfg(true, &["jwt"], &[], false);
    let tok = "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4";
    let r = scrub(&format!("jwt={tok} end"), &c);
    assert_eq!(r.match_count, 1);
    assert!(r.content.contains("[REDACTED:jwt]"));
}

#[test]
fn aws_keys_group() {
    let c = cfg(true, &["aws_keys"], &[], false);
    let r = scrub("key AKIAIOSFODNN7EXAMPLE end", &c);
    assert_eq!(r.match_count, 1);
    assert!(r.content.contains("[REDACTED:aws_keys]"));
}

/// STS (temporary) credentials start with ASIA, not AKIA. Until 2026-07-31 the
/// group matched only AKIA, so pasted S3 pre-signed URLs stored their key id,
/// signature and session token in plaintext. Mirrors
/// tests/test_redaction_parity.py::test_aws_sts_key_is_redacted.
///
/// Fixtures below are SYNTHETIC (sequential/repeated filler) — never paste a
/// real credential into a test, even an expired one: it would publish the
/// secret and permanently record the leak in git history.
const STS_ID: &str = "ASIAEXAMPLE1234567XZ";
const STS_SIG: &str = "c2lnbmF0dXJlRXhhbXBsZTAwMDA%3D";
const STS_TOKEN: &str =
    "FwoGZXIvYXdzEExampleSessionTokenPaddingAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

#[test]
fn aws_sts_temporary_key_is_redacted() {
    let c = cfg(true, &["aws_keys"], &[], false);
    let r = scrub(&format!("id={STS_ID} end"), &c);
    assert!(r.match_count > 0, "STS key id left in plaintext");
    assert!(!r.content.contains(STS_ID));
}

#[test]
fn aws_presigned_url_fully_redacted() {
    let c = cfg(true, &["aws_keys"], &[], false);
    let url = format!(
        "https://x.s3.amazonaws.com/f.md\
         ?AWSAccessKeyId={STS_ID}\
         &Signature={STS_SIG}\
         &x-amz-security-token={STS_TOKEN}"
    );
    let r = scrub(&url, &c);
    assert!(!r.content.contains(STS_ID), "key id survived");
    assert!(!r.content.contains(STS_SIG), "signature survived");
    assert!(!r.content.contains(STS_TOKEN), "session token survived");
}

/// Widening must not turn ordinary English into [REDACTED].
#[test]
fn aws_patterns_do_not_eat_prose() {
    let c = cfg(true, &["aws_keys"], &[], false);
    for prose in [
        "We discussed the AWS migration and the signature of the function.",
        "ASIA is a continent; AKIA Corp makes batteries.",
        "The method signature changed in this release.",
    ] {
        let r = scrub(prose, &c);
        assert_eq!(r.match_count, 0, "false positive on prose: {}", r.content);
    }
}

#[test]
fn custom_regex_group() {
    let c = cfg(true, &["custom_regex"], &[r"MY_SECRET_\d+"], false);
    let r = scrub("MY_SECRET_123 and MY_SECRET_456", &c);
    assert_eq!(r.match_count, 2);
    assert!(r.groups_fired.contains(&"custom_regex".to_string()));
}

#[test]
fn bad_custom_regex_does_not_crash() {
    let c = cfg(true, &["custom_regex"], &["[unclosed"], false);
    let red = Redactor::new(&c);
    assert!(!red.compile_errors().is_empty(), "expected a compile error");
    let r = red.apply("irrelevant");
    assert_eq!(r.match_count, 0);
    assert!(!r.compile_errors.is_empty());
}

#[test]
fn pii_gating_off_by_default() {
    // "pii" in patterns but redact_pii false -> not active.
    let c = cfg(true, &["pii"], &[], false);
    let r = scrub("reach me at alice@example.com", &c);
    assert_eq!(r.match_count, 0);
    assert!(!r.content.contains("[REDACTED:pii]"));
}

#[test]
fn pii_gating_on() {
    let c = cfg(true, &["pii"], &[], true);
    let r = scrub("reach me at alice@example.com", &c);
    assert_eq!(r.match_count, 1);
    assert!(r.content.contains("[REDACTED:pii]"));
}

#[test]
fn pii_needs_pattern_entry_too() {
    // redact_pii true but "pii" not in patterns -> not active.
    let c = cfg(true, &["api_keys"], &[], true);
    let r = scrub("reach me at alice@example.com", &c);
    assert_eq!(r.match_count, 0);
}

#[test]
fn evaluation_order_and_groups_fired() {
    let c = cfg(true, &["api_keys", "github_tokens", "pii"], &[], true);
    let input = format!(
        "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ghp_{} a@b.com",
        "a".repeat(36)
    );
    let r = scrub(&input, &c);
    assert_eq!(r.groups_fired, vec!["api_keys", "github_tokens", "pii"]);
    assert_eq!(r.match_count, 3);
}

#[test]
fn eval_order_sensitive_input() {
    // openai_generic (sk-[A-Za-z0-9]{20,}) would also match an sk-ant key's
    // tail, but anthropic runs first within api_keys and consumes it.
    let c = cfg(true, &["api_keys"], &[], false);
    let r = scrub("sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", &c);
    assert_eq!(r.match_count, 1);
    assert_eq!(r.content, "[REDACTED:api_keys]");
}

#[test]
fn clean_text_unchanged() {
    let c = cfg(true, &["api_keys", "github_tokens"], &[], false);
    let input = "the quick brown fox jumps over the lazy dog";
    let r = scrub(input, &c);
    assert_eq!(r.content, input);
    assert_eq!(r.match_count, 0);
    assert!(r.groups_fired.is_empty());
}

#[test]
fn empty_string() {
    let c = cfg(true, &["api_keys"], &[], false);
    let r = scrub("", &c);
    assert_eq!(r.content, "");
    assert_eq!(r.match_count, 0);
}

#[test]
fn redactor_reusable() {
    let c = cfg(true, &["aws_keys"], &[], false);
    let red = Redactor::new(&c);
    let r1 = red.apply("AKIAIOSFODNN7EXAMPLE");
    let r2 = red.apply("nothing here");
    assert_eq!(r1.match_count, 1);
    assert_eq!(r2.match_count, 0);
}
