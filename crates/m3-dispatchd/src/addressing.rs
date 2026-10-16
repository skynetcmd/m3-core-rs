//! Which inbox reads see a message addressed to a recipient.
//!
//! m3's Python renders this rule as SQL (`orchestration._addressing_predicate`);
//! this is the same rule as a function, so the daemon wakes exactly the readers
//! whose inbox read would return the message. A parity test on the Python side
//! checks the two agree, so neither can drift alone.
//!
//! * A QUALIFIED reader (`type@instance`) sees exactly its own id.
//! * A BARE reader (`type`) sees its own id and every `type@...` instance: a
//!   fan-out read. An id with an empty instance (`type@`) counts as bare, as
//!   Python's `split_agent_id` treats it.

/// The separator between an agent type and its instance.
pub const AGENT_SEP: char = '@';

/// True if `agent_id` names one instance (`type@instance`, instance non-empty).
pub fn is_qualified(agent_id: &str) -> bool {
    matches!(agent_id.split_once(AGENT_SEP), Some((_, tail)) if !tail.is_empty())
}

/// True if an inbox read by `reader` returns a message addressed to `recipient`.
pub fn sees(reader: &str, recipient: &str) -> bool {
    if reader.is_empty() {
        return false;
    }
    if reader == recipient {
        return true;
    }
    !is_qualified(reader)
        && recipient.len() > reader.len()
        && recipient.starts_with(reader)
        && recipient[reader.len()..].starts_with(AGENT_SEP)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rule() {
        assert!(sees("claude-code", "claude-code"));
        assert!(
            sees("claude-code", "claude-code@abc"),
            "a bare read fans out"
        );
        assert!(sees("claude-code@abc", "claude-code@abc"));
        assert!(
            !sees("claude-code@abc", "claude-code"),
            "a direct read does not see the type inbox"
        );
        assert!(!sees("claude-code@abc", "claude-code@xyz"));
        assert!(
            !sees("claude", "claude-code@abc"),
            "a prefix that is not the whole type"
        );
        assert!(!sees("claude-code", "claude-codex@abc"));
        assert!(sees("a@", "a@@x"), "an empty instance counts as bare");
        assert!(!sees("a@", "a@x"));
        assert!(
            sees("a@b@c", "a@b@c"),
            "the instance may itself contain the separator"
        );
        assert!(!sees("", "anything"));
    }
}
