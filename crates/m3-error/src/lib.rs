//! Shared error type for the m3-core-rs workspace (Phase 1.2).
//!
//! `M3Error` maps directly to Python exceptions in the `m3-core-py` binding layer.

/// The canonical error type returned across the m3-core-rs FFI boundary.
#[derive(Debug)]
pub enum M3Error {
    VectorDimMismatch { expected: usize, got: usize },
    DatabaseLocked,
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
