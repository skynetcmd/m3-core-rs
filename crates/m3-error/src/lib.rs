//! Shared error type for the m3-core-rs workspace (Phase 1.2).
//!
//! `M3Error` maps directly to Python exceptions in the `m3-core-py` binding layer.

/// The canonical error type returned across the m3-core-rs FFI boundary.
#[derive(Debug)]
pub enum M3Error {
    /// `index` is the offending row when the input was a batch, `None` for a
    /// single vector.
    VectorDimMismatch {
        expected: usize,
        got: usize,
        index: Option<usize>,
    },
    DatabaseLocked,
    /// Input exceeds the model's context window — a CLIENT error, distinct from
    /// a backend failure.
    ///
    /// ⚠ This exists as its own variant specifically so the HTTP layer can map
    /// it to 413 while everything else stays 500. It was previously a
    /// `Backend(String)` carrying `"input too long: N tokens > n_ctx M"`, which
    /// the server flattened to an opaque 500 — destroying both numbers before
    /// they left the process and making "your input is too long" indis-
    /// tinguishable from "the server is broken". That is issue #139 repeating
    /// one stack over; the Python server had the identical defect and was fixed
    /// with a structured 413.
    ///
    /// Carrying the counts as FIELDS rather than in a formatted string is the
    /// point: the handler must not have to parse an error message to build a
    /// response. Matching on message text would re-create the same coupling in
    /// a more fragile form.
    ///
    /// `index` is the position of the over-long text in the batch the CALLER
    /// submitted, `None` when unknown.
    InputTooLong {
        tokens: usize,
        n_ctx: usize,
        index: Option<usize>,
    },
    Backend(String),
    Io(std::io::Error),
    Config(String),
    Parity {
        context: String,
        detail: String,
    },
    Other(String),
}

impl std::fmt::Display for M3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            M3Error::VectorDimMismatch {
                expected,
                got,
                index,
            } => {
                write!(
                    f,
                    "vector dimension mismatch: expected {expected}, got {got}"
                )?;
                match index {
                    Some(i) => write!(f, " at row {i}"),
                    None => Ok(()),
                }
            }
            M3Error::DatabaseLocked => write!(f, "database locked"),
            // The leading wording is preserved VERBATIM from the former
            // Backend(String) form: existing clients (and #139's test fixtures)
            // match `N tokens > n_ctx M`. The index is only ever appended.
            M3Error::InputTooLong {
                tokens,
                n_ctx,
                index,
            } => {
                write!(f, "input too long: {tokens} tokens > n_ctx {n_ctx}")?;
                match index {
                    Some(i) => write!(f, " (input {i})"),
                    None => Ok(()),
                }
            }
            M3Error::Backend(m) => write!(f, "backend error: {m}"),
            M3Error::Io(e) => write!(f, "io error: {e}"),
            M3Error::Config(m) => write!(f, "config error: {m}"),
            M3Error::Parity { context, detail } => {
                write!(f, "parity violation in {context}: {detail}")
            }
            M3Error::Other(m) => write!(f, "{m}"),
        }
    }
}

/// One error for each of several waiters (a coalesced batch fails every
/// caller in it). Every variant keeps its type; `Io` is rebuilt from its kind
/// and message, because `std::io::Error` is not `Clone` — the clone loses only
/// the inner error's `source()` chain, never the text or the kind.
impl Clone for M3Error {
    fn clone(&self) -> Self {
        match self {
            M3Error::VectorDimMismatch {
                expected,
                got,
                index,
            } => M3Error::VectorDimMismatch {
                expected: *expected,
                got: *got,
                index: *index,
            },
            M3Error::DatabaseLocked => M3Error::DatabaseLocked,
            M3Error::InputTooLong {
                tokens,
                n_ctx,
                index,
            } => M3Error::InputTooLong {
                tokens: *tokens,
                n_ctx: *n_ctx,
                index: *index,
            },
            M3Error::Backend(m) => M3Error::Backend(m.clone()),
            M3Error::Io(e) => M3Error::Io(std::io::Error::new(e.kind(), e.to_string())),
            M3Error::Config(m) => M3Error::Config(m.clone()),
            M3Error::Parity { context, detail } => M3Error::Parity {
                context: context.clone(),
                detail: detail.clone(),
            },
            M3Error::Other(m) => M3Error::Other(m.clone()),
        }
    }
}

impl std::error::Error for M3Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            M3Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for M3Error {
    fn from(e: std::io::Error) -> Self {
        M3Error::Io(e)
    }
}

/// Workspace-wide result alias.
pub type Result<T> = std::result::Result<T, M3Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_keeps_every_variant_typed() {
        let e = M3Error::Config("bad".into()).clone();
        assert!(matches!(e, M3Error::Config(ref m) if m == "bad"), "{e}");
        let io = M3Error::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone")).clone();
        match io {
            M3Error::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
                assert!(e.to_string().contains("gone"));
            }
            other => panic!("Io became {other:?}"),
        }
    }

    #[test]
    fn input_too_long_keeps_the_parsed_prefix_and_appends_the_index() {
        let e = M3Error::InputTooLong {
            tokens: 9000,
            n_ctx: 8192,
            index: Some(3),
        };
        assert_eq!(
            e.to_string(),
            "input too long: 9000 tokens > n_ctx 8192 (input 3)"
        );
        let e = M3Error::InputTooLong {
            tokens: 9000,
            n_ctx: 8192,
            index: None,
        };
        assert_eq!(e.to_string(), "input too long: 9000 tokens > n_ctx 8192");
    }
}
