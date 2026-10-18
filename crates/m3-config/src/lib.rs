//! m3's two config files, read by the Rust core.
//!
//! m3 has exactly two config files (user decision, m3 memory `9655f30a`):
//!
//! * `m3_config.toml`: every setting, readable. Located by [`roots::locate`]:
//!   `--config <file>` > `M3_CONFIG_ROOT` > `~/.m3/config`.
//! * the secrets file it names (`[secrets] file`, default
//!   `<config root>/m3_secrets.toml`), user-only. [`secrets`] reads one value
//!   from it.
//!
//! The Python owners are m3's `m3_memory/_roots.py` (where the file is),
//! `bin/m3_core/config/` (keys, precedence, `--set`) and
//! `bin/m3_core/secret_store.py` (the secrets file). This crate mirrors them
//! for the processes Python does not run in-process (m3-embed-server), and is
//! the ONE place in the Rust core that locates or parses either file.
//!
//! Precedence for one key, as m3's registry has it:
//! `--set` > m3_config.toml > default. A key's former environment name is not
//! read; it is kept only to name the variable in messages.
//! Every resolved value carries its [`Source`], so a message can name the
//! setting that decided it.

pub mod env;
pub mod os;
pub mod roots;
pub mod secrets;
pub mod security;
pub mod settings;

// Compiled on every OS so its tests run everywhere; only the Windows privacy
// check calls it, so elsewhere it is dead code outside the tests.
#[cfg_attr(not(windows), allow(dead_code))]
mod sddl;
#[cfg(windows)]
mod winacl;

pub use env::{Env, MapEnv, ProcessEnv};
pub use roots::{locate, Location, Tier, CONFIG_FILE_NAME, CONFIG_ROOT_ENV, MEMORY_ROOT_ENV};
pub use settings::{
    parse_bool, parse_set, Config, ConfigError, Key, Kind, Overrides, Source, Value,
};
