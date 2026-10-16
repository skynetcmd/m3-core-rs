//! The secrets file: the minimum the Rust core needs, read the way m3's owner
//! (`bin/m3_core/secret_store.py`) reads it.
//!
//! * Where: `[secrets] file` in m3_config.toml (relative to the config root),
//!   else `<config root>/m3_secrets.toml`.
//! * Format: one `[secrets]` table of `NAME = "value"` or
//!   `NAME = { keyring = "service/key" }`.
//! * Refused before it is parsed when anyone but the user can read it: on Unix
//!   a group/other mode bit or another owner; on Windows a DACL admitting anyone
//!   beyond the user, SYSTEM, Administrators and OWNER RIGHTS. Read as a Windows
//!   service (LocalSystem), the file's owner counts as the user: the service's
//!   secrets file is necessarily another account's.
//!
//! Not ported, deliberately: the keyring (a value stored there is refused with
//! the command that moves it into the file) and the check that every name in
//! the file is a known m3 secret (m3 validates its own file; this reads one
//! entry). A message never shows a secret's value.

use std::path::{Path, PathBuf};

use crate::settings::{Config, ConfigError, Key, Source};

/// The registry key naming the secrets file (m3 `keys.SECRETS_FILE`): a
/// locator, so never a `--set` flag.
pub const SECRETS_FILE: Key = Key::path("secrets.file", &[]).locator();
pub const SECRETS_FILE_NAME: &str = "m3_secrets.toml";

/// The secrets file `cfg` names, and where that came from.
pub fn secrets_file(cfg: &Config) -> Result<(PathBuf, Source), ConfigError> {
    match cfg.path(&SECRETS_FILE)? {
        Some((p, from)) => Ok((cfg.under_root(&p), from)),
        None => Ok((cfg.location().root.join(SECRETS_FILE_NAME), Source::Default)),
    }
}

/// The value of `name` in the secrets file at `path`; `None` when the file or
/// the entry is absent. A file that exists but is not private, not readable or
/// malformed is an error, never "absent".
pub fn read_secret(path: &Path, name: &str) -> Result<Option<String>, ConfigError> {
    if !path.is_file() {
        return Ok(None);
    }
    check_private(path)?;
    let text = std::fs::read_to_string(path).map_err(|e| {
        ConfigError(format!(
            "cannot read the secrets file {} ({e})",
            path.display()
        ))
    })?;
    let doc: toml::Table = toml::from_str(&text).map_err(|e| {
        // The parser quotes the line; it may hold a value, so give the reason only.
        ConfigError(format!(
            "the secrets file {} is not valid TOML ({}); fix the line, or remove the file",
            path.display(),
            e.message()
        ))
    })?;
    let table = match doc.get("secrets") {
        None => return Ok(None),
        Some(toml::Value::Table(t)) => t,
        Some(_) => {
            return Err(ConfigError(format!(
                "{}: 'secrets' must be a [secrets] table",
                path.display()
            )))
        }
    };
    match table.get(name) {
        None => Ok(None),
        Some(toml::Value::String(v)) => Ok(Some(v.trim().to_string())),
        Some(toml::Value::Table(t)) if t.len() == 1 && t.get("keyring").is_some_and(|k| k.is_str()) => {
            Err(ConfigError(format!(
                "{name} in the secrets file {} is stored in the OS keyring ({}), which this program \
                 cannot read. Store the value in the file instead: m3 secret set {name}",
                path.display(),
                t["keyring"].as_str().unwrap_or_default()
            )))
        }
        Some(_) => Err(ConfigError(format!(
            "{}: [secrets] {name} must be NAME = \"value\" or NAME = {{ keyring = \"service/key\" }}",
            path.display()
        ))),
    }
}

#[cfg(unix)]
fn check_private(p: &Path) -> Result<(), ConfigError> {
    use std::os::unix::fs::MetadataExt;
    let st = std::fs::metadata(p).map_err(|e| {
        ConfigError(format!(
            "cannot inspect the secrets file {} ({e})",
            p.display()
        ))
    })?;
    let mode = st.mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(ConfigError(format!(
            "refusing to read the secrets file {}: it is mode {mode:#o}, so others can read it; a \
             secrets file must be user-only. Fix: chmod 600 {}",
            p.display(),
            p.display()
        )));
    }
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    if st.uid() != uid {
        return Err(ConfigError(format!(
            "refusing to read the secrets file {}: it is owned by uid {}, not this process's uid {uid}. \
             Fix: if you created it, `chown $(id -u) {}` as that owner or root; otherwise remove it",
            p.display(),
            st.uid(),
            p.display()
        )));
    }
    Ok(())
}

#[cfg(windows)]
fn check_private(p: &Path) -> Result<(), ConfigError> {
    let refuse = |why: String| {
        ConfigError(format!(
            "refusing to read the secrets file {}: cannot verify who can read it ({why})",
            p.display()
        ))
    };
    let user = crate::winacl::current_user_sid().map_err(refuse)?;
    let sddl = crate::winacl::file_sddl(p).map_err(refuse)?;
    let owner = if user == crate::sddl::SYSTEM_SID {
        crate::sddl::owner(&sddl)
    } else {
        None
    };
    let foreign = crate::sddl::foreign_principals(&sddl, &user, owner);
    if foreign.is_empty() {
        return Ok(());
    }
    // m3's `win_acl.icacls_user_only`, with the user spelled out. As a service
    // the account is not the user's, so the fix names the owner generically.
    let fix = match std::env::var("USERNAME") {
        Ok(u) if user != crate::sddl::SYSTEM_SID && !u.is_empty() => format!(
            "icacls \"{}\" /inheritance:r /grant:r \"{u}:F\" /grant:r \"SYSTEM:F\"",
            p.display()
        ),
        _ => format!(
            "as the file's owner: icacls \"{}\" /inheritance:r /grant:r \"<owner>:F\" /grant:r \"SYSTEM:F\"",
            p.display()
        ),
    };
    Err(ConfigError(format!(
        "refusing to read the secrets file {}: it can be opened by {} besides its user; a secrets file \
         must be user-only. Fix: {fix}",
        p.display(),
        foreign.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::roots::{Location, Tier};
    use crate::settings::Overrides;

    fn cfg_in(dir: &Path, text: &str) -> Config {
        let file = dir.join("m3_config.toml");
        std::fs::write(&file, text).unwrap();
        let loc = Location {
            file,
            root: dir.to_path_buf(),
            tier: Tier::Arg,
        };
        Config::load(
            loc,
            Overrides::default(),
            Some(Box::new(MapEnv::new())),
            None,
        )
        .unwrap()
    }

    /// Owner-only, as m3's `atomic_write` creates it.
    fn write_private(p: &Path, text: &str) {
        std::fs::write(p, text).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
    }

    #[test]
    fn the_default_is_beside_m3_config_and_a_relative_file_is_under_the_root() {
        let d = tempfile::tempdir().unwrap();
        let (p, from) = secrets_file(&cfg_in(d.path(), "")).unwrap();
        assert_eq!(
            (p, from),
            (d.path().join("m3_secrets.toml"), Source::Default)
        );
        let (p, from) =
            secrets_file(&cfg_in(d.path(), "[secrets]\nfile = \"private/s.toml\"\n")).unwrap();
        assert_eq!(p, d.path().join("private").join("s.toml"));
        assert!(from.to_string().starts_with("[secrets].file in "), "{from}");
    }

    #[test]
    fn the_value_is_read_from_the_named_file() {
        let d = tempfile::tempdir().unwrap();
        let s = d.path().join("elsewhere.toml");
        write_private(
            &s,
            "[secrets]\nM3_EMBED_SERVER_TOKEN_SHA256 = \"abc\"\nOTHER = { keyring = \"a/b\" }\n",
        );
        let cfg = cfg_in(
            d.path(),
            &format!("[secrets]\nfile = {:?}\n", s.to_string_lossy()),
        );
        let (path, _) = secrets_file(&cfg).unwrap();
        assert_eq!(
            read_secret(&path, "M3_EMBED_SERVER_TOKEN_SHA256")
                .unwrap()
                .as_deref(),
            Some("abc")
        );
        assert_eq!(read_secret(&path, "MISSING").unwrap(), None);
        let e = read_secret(&path, "OTHER").unwrap_err().0;
        assert!(
            e.contains("OS keyring") && e.contains("m3 secret set OTHER"),
            "{e}"
        );
        assert_eq!(
            read_secret(&d.path().join("absent.toml"), "X").unwrap(),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_group_or_world_readable_file_is_refused_without_its_value() {
        use std::os::unix::fs::PermissionsExt;
        let d = tempfile::tempdir().unwrap();
        let s = d.path().join("m3_secrets.toml");
        for mode in [0o640, 0o604, 0o644] {
            std::fs::write(
                &s,
                "[secrets]\nM3_EMBED_SERVER_TOKEN_SHA256 = \"deadbeef\"\n",
            )
            .unwrap();
            std::fs::set_permissions(&s, std::fs::Permissions::from_mode(mode)).unwrap();
            let e = read_secret(&s, "M3_EMBED_SERVER_TOKEN_SHA256")
                .unwrap_err()
                .0;
            assert!(e.contains("user-only") && e.contains("chmod 600"), "{e}");
            assert!(!e.contains("deadbeef"), "the value leaked: {e}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_file_others_can_open_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let s = d.path().join("m3_secrets.toml");
        std::fs::write(&s, "[secrets]\nX = \"v\"\n").unwrap();
        // Grant Everyone read access; the check must see it.
        let ok = std::process::Command::new("icacls")
            .arg(&s)
            .args(["/grant", "*S-1-1-0:R"])
            .output()
            .unwrap();
        assert!(
            ok.status.success(),
            "{}",
            String::from_utf8_lossy(&ok.stderr)
        );
        let e = read_secret(&s, "X").unwrap_err().0;
        assert!(e.contains("Everyone") && e.contains("user-only"), "{e}");
    }
}
