//! Shared error type for the m3-core-rs workspace (Phase 1.2).
//!
//! `M3Error` maps directly to Python exceptions in the `m3-core-py` binding layer.

/// The canonical error type returned across the m3-core-rs FFI boundary.
#[derive(Debug)]
pub enum M3Error {
    VectorDimMismatch { expected: usize, got: usize },
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
    InputTooLong { tokens: usize, n_ctx: usize },
    Backend(String),
    Io(std::io::Error),
    Config(String),
    Parity { context: String, detail: String },
    Other(String),
}

impl std::fmt::Display for M3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            M3Error::VectorDimMismatch { expected, got } => {
                write!(f, "vector dimension mismatch: expected {expected}, got {got}")
            }
            M3Error::DatabaseLocked => write!(f, "database locked"),
            // Wording preserved VERBATIM from the former Backend(String) form:
            // existing clients (and #139's test fixtures) parse this text.
            M3Error::InputTooLong { tokens, n_ctx } => {
                write!(f, "input too long: {tokens} tokens > n_ctx {n_ctx}")
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
