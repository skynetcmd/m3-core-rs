//! Where environment values and the home directory come from.
//!
//! Behind a trait so tests resolve against a map instead of the process
//! environment: tests share one process, and a variable set by one test (or by
//! the developer's shell) must not decide another's result.

use std::collections::HashMap;
use std::path::PathBuf;

/// The environment a resolution reads.
pub trait Env: Send + Sync {
    /// The value of `name`, or `None` when it is unset or empty. An empty value
    /// counts as unset, as in m3's `_roots._env` (`os.environ.get(name) or None`).
    fn var(&self, name: &str) -> Option<String>;
    /// The user's home directory, as Python's `os.path.expanduser("~")` finds it.
    fn home(&self) -> Option<PathBuf>;
    /// The directory a relative root is made absolute against.
    fn cwd(&self) -> Option<PathBuf>;
}

/// The real process environment.
pub struct ProcessEnv;

impl Env for ProcessEnv {
    fn var(&self, name: &str) -> Option<String> {
        match std::env::var(name) {
            Ok(v) if !v.is_empty() => Some(v),
            Ok(_) | Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(raw)) => {
                log::warn!(
                    "ignoring the environment variable {name}: its value {raw:?} is not valid UTF-8; \
                     unset it or set it to a valid path or value"
                );
                None
            }
        }
    }

    fn home(&self) -> Option<PathBuf> {
        // USERPROFILE on Windows and HOME on Unix, each with the OS's own
        // fallback: the same sources Python's expanduser("~") reads.
        std::env::home_dir().filter(|p| !p.as_os_str().is_empty())
    }

    fn cwd(&self) -> Option<PathBuf> {
        std::env::current_dir().ok()
    }
}

/// A fixed environment: tests, and resolutions that must not see the process
/// environment.
#[derive(Debug, Clone, Default)]
pub struct MapEnv {
    pub vars: HashMap<String, String>,
    pub home: Option<PathBuf>,
    pub cwd: Option<PathBuf>,
}

impl MapEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, name: &str, value: &str) -> Self {
        self.vars.insert(name.to_string(), value.to_string());
        self
    }

    pub fn with_home(mut self, home: impl Into<PathBuf>) -> Self {
        self.home = Some(home.into());
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

impl Env for MapEnv {
    fn var(&self, name: &str) -> Option<String> {
        self.vars.get(name).filter(|v| !v.is_empty()).cloned()
    }

    fn home(&self) -> Option<PathBuf> {
        self.home.clone()
    }

    fn cwd(&self) -> Option<PathBuf> {
        self.cwd.clone()
    }
}
