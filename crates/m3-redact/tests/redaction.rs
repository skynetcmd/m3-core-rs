//! Per-pattern redaction coverage. Stands in for the pending Python-parity
//! corpus test (corpus not yet available in this workspace).

use m3_redact::{redact, RedactionProfile};

fn p() -> RedactionProfile {
    RedactionProfile::default()
}

#[test]
fn aws_access_key_positive() {
    let r = redact("key is AKIAIOSFODNN7EXAMPLE done", &p());
    assert!(r.text.contains("[REDACTED:aws_access_key]"));
    assert_eq!(r.hits, 1);
    assert_eq!(r.breakdown.get("aws_access_key"), Some(&1));
}

#[test]
fn aws_access_key_negative() {
    let r = redact("AKIA is too short", &p());
    assert_eq!(r.hits, 0);
}

#[test]
fn anthropic_key_positive() {
    let r = redact("sk-ant-api03-AbCdEf0123456789AbCdEf done", &p());
    assert!(r.text.contains("[REDACTED:anthropic_key]"));
    assert_eq!(r.hits, 1);
}

#[test]
fn openai_key_positive_and_anthropic_not_misclassified() {
    let generic = redact("token sk-AbCdEf0123456789AbCdEfGh end", &p());
    assert!(generic.text.contains("[REDACTED:openai_key]"));
    let anth = redact("sk-ant-abcdefghij0123456789kl end", &p());
    assert!(anth.text.contains("[REDACTED:anthropic_key]"));
    assert!(!anth.text.contains("[REDACTED:openai_key]"));
}

#[test]
fn google_api_key_positive() {
    let r = redact("g AIzaSyA1234567890abcdefABCDEF_ghijk-lmn x", &p());
    assert!(r.text.contains("[REDACTED:google_api_key]"));
}

#[test]
fn github_token_positive() {
    let r = redact("ghp_0123456789abcdefABCDEF0123456789abcd here", &p());
    assert!(r.text.contains("[REDACTED:github_token]"));
}

#[test]
fn email_positive_negative() {
    let r = redact("reach me at alice@example.com please", &p());
    assert!(r.text.contains("[REDACTED:email]"));
    let neg = redact("not an email: foo@bar", &p());
    assert_eq!(neg.hits, 0);
}

#[test]
fn ipv4_positive_negative() {
    let r = redact("host 192.168.1.42 listening", &p());
    assert!(r.text.contains("[REDACTED:ipv4]"));
    let neg = redact("version 999.999.999.999 nope", &p());
    assert_eq!(neg.hits, 0);
}

#[test]
fn fixed_prefix_scan() {
    let r = redact("slack xoxb-123456789012-abcdefABCDEF token", &p());
    assert!(r.text.contains("[REDACTED:known_prefix]"));
    assert_eq!(r.breakdown.get("known_prefix"), Some(&1));
}

#[test]
fn multiple_kinds_breakdown() {
    let input = "mail alice@example.com ip 10.0.0.1 key AKIAIOSFODNN7EXAMPLE";
    let r = redact(input, &p());
    assert_eq!(r.hits, 3);
    assert_eq!(r.breakdown.get("email"), Some(&1));
    assert_eq!(r.breakdown.get("ipv4"), Some(&1));
    assert_eq!(r.breakdown.get("aws_access_key"), Some(&1));
}

#[test]
fn clean_text_unchanged() {
    let input = "the quick brown fox jumps over the lazy dog";
    let r = redact(input, &p());
    assert_eq!(r.text, input);
    assert_eq!(r.hits, 0);
}

#[test]
fn idempotent_redaction() {
    let input = "mail alice@example.com ip 10.0.0.1 key AKIAIOSFODNN7EXAMPLE";
    let once = redact(input, &p());
    let twice = redact(&once.text, &p());
    assert_eq!(once.text, twice.text);
    assert_eq!(twice.hits, 0, "markers must not themselves match patterns");
}
