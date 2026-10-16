//! The security seam's Rust owner: the FIPS tier and the keys that set it.
//!
//! m3's seam hierarchy is version < OS < security < database < logic (user
//! decision, m3 memory `f5aecb25`). The Python owner is m3's
//! `bin/crypto_provider.py` (`fips_tier()`); this module is the ONE place in the
//! Rust core that names `crypto.fips_mode` / `crypto.fips_strict` or decides the
//! tier (a test in this crate fails if another crate names them). The strings
//! are Python's, so a parity test can compare the two languages.

use crate::settings::{Config, ConfigError, Key, Source};

pub const FIPS_MODE: Key = Key::bool("crypto.fips_mode", &["M3_FIPS_MODE"]);
pub const FIPS_STRICT: Key = Key::bool("crypto.fips_strict", &["M3_FIPS_STRICT"]);

/// The FIPS tier, as `crypto_provider.fips_tier()` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FipsTier {
    Off,
    /// FIPS-approved algorithms through the FIPS-ready backend.
    Mode,
    /// Also require a validated module; anything that cannot prove one is refused.
    Strict,
}

impl FipsTier {
    pub fn as_str(self) -> &'static str {
        match self {
            FipsTier::Off => "off",
            FipsTier::Mode => "mode",
            FipsTier::Strict => "strict",
        }
    }

    /// Mode or strict: the FIPS-ready backend is required.
    pub fn is_fips(self) -> bool {
        self != FipsTier::Off
    }
}

/// The tier and where each of its two settings came from (for messages that
/// name the setting to change).
#[derive(Debug, Clone)]
pub struct FipsSetting {
    pub tier: FipsTier,
    pub mode_source: Source,
    pub strict_source: Source,
}

/// THE tier rule, as Python's: strict if `crypto.fips_strict`, else mode if
/// `crypto.fips_mode`, else off (strict implies mode).
pub fn fips_tier(cfg: &Config) -> Result<FipsSetting, ConfigError> {
    let (mode, mode_source) = cfg.bool_or(&FIPS_MODE, false)?;
    let (strict, strict_source) = cfg.bool_or(&FIPS_STRICT, false)?;
    let tier = if strict {
        FipsTier::Strict
    } else if mode {
        FipsTier::Mode
    } else {
        FipsTier::Off
    };
    Ok(FipsSetting {
        tier,
        mode_source,
        strict_source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::roots::{Location, Tier};
    use crate::settings::Overrides;

    fn cfg(toml: &str, env: MapEnv) -> Config {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("m3_config.toml");
        std::fs::write(&file, toml).unwrap();
        let loc = Location {
            file,
            root: dir.path().to_path_buf(),
            tier: Tier::Arg,
        };
        let c = Config::load(loc, Overrides::default(), Some(Box::new(env)), None).unwrap();
        drop(dir);
        c
    }

    #[test]
    fn the_tier_rule_matches_python() {
        let cases = [
            ("", FipsTier::Off),
            ("[crypto]\nfips_mode = true\n", FipsTier::Mode),
            ("[crypto]\nfips_strict = true\n", FipsTier::Strict),
            (
                "[crypto]\nfips_mode = true\nfips_strict = true\n",
                FipsTier::Strict,
            ),
            (
                "[crypto]\nfips_mode = false\nfips_strict = false\n",
                FipsTier::Off,
            ),
        ];
        for (toml, want) in cases {
            let got = fips_tier(&cfg(toml, MapEnv::new())).unwrap().tier;
            assert_eq!(got, want, "{toml:?}");
        }
        assert_eq!(FipsTier::Strict.as_str(), "strict");
        assert!(FipsTier::Mode.is_fips() && !FipsTier::Off.is_fips());
    }

    #[test]
    fn the_file_outranks_the_legacy_env_name() {
        let env = MapEnv::new().with("M3_FIPS_MODE", "0");
        let s = fips_tier(&cfg("[crypto]\nfips_mode = true\n", env)).unwrap();
        assert_eq!(s.tier, FipsTier::Mode);
        assert!(
            matches!(s.mode_source, Source::File { .. }),
            "{:?}",
            s.mode_source
        );
    }

    /// The seam's gate: no other crate names a FIPS key. A second reader is how
    /// FIPS from m3_config.toml went missing (the embed server read its own file,
    /// m3 Python read env alone).
    #[test]
    fn only_this_module_names_the_fips_keys() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let mut offenders = Vec::new();
        let mut stack = vec![crates.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let p = entry.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if p.is_dir() {
                    if name != "target" && !name.starts_with('.') {
                        stack.push(p);
                    }
                    continue;
                }
                // This crate owns the keys (its other modules' tests use one as
                // a sample flag); the rule is about every OTHER crate.
                let in_owner = p.components().any(|c| c.as_os_str() == "m3-config");
                if p.extension().and_then(|e| e.to_str()) != Some("rs") || in_owner {
                    continue;
                }
                let text = std::fs::read_to_string(&p).unwrap_or_default();
                for needle in ["\"crypto.fips_mode\"", "\"crypto.fips_strict\""] {
                    if text.contains(needle) {
                        offenders.push(format!("{} names {needle}", p.display()));
                    }
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "the FIPS keys belong to m3_config::security (fips_tier, FIPS_MODE, FIPS_STRICT); \
             import them instead:\n  {}",
            offenders.join("\n  ")
        );
    }
}
