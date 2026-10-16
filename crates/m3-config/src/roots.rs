//! Where m3_config.toml is.
//!
//! Mirrors m3's owner, `m3_memory/_roots.py` (`config_root_with_tier`,
//! `_derived`, `abs_root`) and `bin/m3_core/config/loader.py` (`locate`):
//!
//! `--config <file>` > `M3_CONFIG_ROOT` > `M3_MEMORY_ROOT/config` (legacy)
//! > `~/.m3/config`; the file is `<root>/m3_config.toml`.
//!
//! The two languages cannot share the function, so the cases are pinned by the
//! tests below; a change to the Python precedence must change this file too.

use std::path::{Component, Path, PathBuf};

use crate::env::Env;

pub const CONFIG_ROOT_ENV: &str = "M3_CONFIG_ROOT";
pub const MEMORY_ROOT_ENV: &str = "M3_MEMORY_ROOT";
pub const CONFIG_FILE_NAME: &str = "m3_config.toml";

/// Which precedence tier located the file. `as_str` spells it as m3 does
/// (`_roots.CONFIG_ROOT_*`), so `m3 config show --origin` and this server's
/// log name a tier the same way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Arg,
    Env,
    Legacy,
    Default,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Arg => "arg:--config",
            Tier::Env => "env:M3_CONFIG_ROOT",
            Tier::Legacy => "legacy:M3_MEMORY_ROOT/config",
            Tier::Default => "default:~/.m3/config",
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The config file, its root (the directory relative paths in it are against)
/// and the tier that answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub file: PathBuf,
    pub root: PathBuf,
    pub tier: Tier,
}

/// `explicit` is the `--config` FILE (made absolute here; its parent is the
/// root). Without it the environment tiers decide. Errors only when no tier can
/// answer (no home directory and no root variable).
pub fn locate(explicit: Option<&Path>, env: &dyn Env) -> Result<Location, String> {
    if let Some(f) = explicit {
        let f = expand_user(&f.to_string_lossy(), env);
        let f = absolute(Path::new(&f), env)?;
        let root = f
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("--config {} has no parent directory", f.display()))?;
        return Ok(Location {
            file: f,
            root,
            tier: Tier::Arg,
        });
    }
    let (root, tier) = config_root_with_tier(env)?;
    if tier == Tier::Legacy {
        log::warn!(
            "the m3 config root comes from the legacy M3_MEMORY_ROOT/config tier ({}), which m3 \
             removes at N+3. Set M3_CONFIG_ROOT, or pass --config <file>",
            root.display()
        );
    }
    // One separator style for display (`abs_root` returns forward slashes).
    let root: PathBuf = root.components().collect();
    Ok(Location {
        file: root.join(CONFIG_FILE_NAME),
        root,
        tier,
    })
}

/// `_roots.config_root_with_tier` without the process pin (that is `--config`,
/// handled by [`locate`]).
pub fn config_root_with_tier(env: &dyn Env) -> Result<(PathBuf, Tier), String> {
    if let Some(root) = env.var(CONFIG_ROOT_ENV) {
        return Ok((PathBuf::from(abs_root(&root, env)?), Tier::Env));
    }
    if let Some(mem) = env.var(MEMORY_ROOT_ENV) {
        return Ok((
            PathBuf::from(format!("{}/config", abs_root(&mem, env)?)),
            Tier::Legacy,
        ));
    }
    let home = env.home().ok_or_else(|| {
        "cannot locate m3_config.toml: no home directory could be determined (HOME / USERPROFILE) \
         and neither M3_CONFIG_ROOT nor M3_MEMORY_ROOT is set. Pass --config <file>, or set \
         M3_CONFIG_ROOT"
            .to_string()
    })?;
    Ok((home.join(".m3").join("config"), Tier::Default))
}

/// `_roots.abs_root`: expand `~`, and make a relative root absolute WITHOUT
/// rewriting one that is already absolute under either path convention (a
/// POSIX root exported from Git Bash or WSL must not become `C:\data\.m3`).
/// The result uses forward slashes, as Python's does.
pub fn abs_root(root: &str, env: &dyn Env) -> Result<String, String> {
    let p = expand_user(root, env);
    if is_posix_abs(&p) || is_nt_abs(&p) {
        return Ok(p.replace('\\', "/"));
    }
    Ok(absolute(Path::new(&p), env)?
        .to_string_lossy()
        .replace('\\', "/"))
}

/// `posixpath.isabs`.
fn is_posix_abs(p: &str) -> bool {
    p.starts_with('/')
}

/// `ntpath.isabs` as Python 3.13+ has it: a drive with a root (`C:\`, `C:/`) or
/// a UNC prefix (`\\`, `//`). (3.12 also took a lone leading `\`.)
fn is_nt_abs(p: &str) -> bool {
    let b = p.as_bytes();
    let drive_root = b.len() >= 3
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b[2] == b'\\' || b[2] == b'/');
    drive_root || p.starts_with("\\\\") || p.starts_with("//")
}

/// `os.path.expanduser` for `~` and `~/...` (either separator). `~user` is
/// left as it is. Without a home the text is returned unchanged, as Python does.
pub fn expand_user(p: &str, env: &dyn Env) -> String {
    let rest = if p == "~" {
        Some("")
    } else {
        p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\"))
    };
    match (rest, env.home()) {
        (Some(""), Some(home)) => home.to_string_lossy().into_owned(),
        (Some(rest), Some(home)) => home.join(rest).to_string_lossy().into_owned(),
        _ => p.to_string(),
    }
}

/// `os.path.abspath`: joined to the working directory and normalised
/// lexically (`.` dropped, `..` pops), never touching the filesystem.
fn absolute(p: &Path, env: &dyn Env) -> Result<PathBuf, String> {
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        let cwd = env.cwd().ok_or_else(|| {
            format!(
                "cannot make {} absolute: the working directory is unknown; give an absolute path",
                p.display()
            )
        })?;
        cwd.join(p)
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! The cases of m3's config-root precedence (`_roots.config_root_with_tier`,
    //! `loader.locate`), each against a map environment: never the real one.
    use super::*;
    use crate::env::MapEnv;

    fn s(p: &Path) -> String {
        p.to_string_lossy().replace('\\', "/")
    }

    fn base() -> MapEnv {
        MapEnv::new().with_home("/h/user").with_cwd("/work/dir")
    }

    #[test]
    fn config_root_env_wins_over_memory_root_and_default() {
        let env = base()
            .with(CONFIG_ROOT_ENV, "/srv/m3cfg")
            .with(MEMORY_ROOT_ENV, "/srv/mem");
        let loc = locate(None, &env).unwrap();
        assert_eq!(loc.tier, Tier::Env);
        assert_eq!(s(&loc.root), "/srv/m3cfg");
        assert_eq!(s(&loc.file), "/srv/m3cfg/m3_config.toml");
    }

    #[test]
    fn memory_root_gives_its_config_subdirectory_as_the_legacy_tier() {
        let env = base().with(MEMORY_ROOT_ENV, "/srv/mem");
        let loc = locate(None, &env).unwrap();
        assert_eq!(loc.tier, Tier::Legacy);
        assert_eq!(s(&loc.root), "/srv/mem/config");
    }

    #[test]
    fn nothing_set_is_home_dot_m3_config() {
        let loc = locate(None, &base()).unwrap();
        assert_eq!(loc.tier, Tier::Default);
        assert_eq!(s(&loc.file), "/h/user/.m3/config/m3_config.toml");
    }

    #[test]
    fn an_empty_variable_counts_as_unset() {
        let env = base()
            .with(CONFIG_ROOT_ENV, "")
            .with(MEMORY_ROOT_ENV, "/srv/mem");
        assert_eq!(locate(None, &env).unwrap().tier, Tier::Legacy);
    }

    #[test]
    fn the_explicit_file_outranks_every_variable() {
        let env = base().with(CONFIG_ROOT_ENV, "/srv/m3cfg");
        let loc = locate(Some(Path::new("/pinned/c.toml")), &env).unwrap();
        assert_eq!(loc.tier, Tier::Arg);
        assert_eq!(s(&loc.file), "/pinned/c.toml");
        assert_eq!(s(&loc.root), "/pinned");
    }

    #[test]
    fn a_relative_explicit_file_is_made_absolute() {
        let loc = locate(Some(Path::new("sub/../c.toml")), &base()).unwrap();
        assert_eq!(s(&loc.file), "/work/dir/c.toml");
    }

    #[test]
    fn tilde_and_relative_roots_are_expanded_and_made_absolute() {
        let env = base().with(CONFIG_ROOT_ENV, "~/cfg");
        assert_eq!(s(&locate(None, &env).unwrap().root), "/h/user/cfg");
        let env = base().with(CONFIG_ROOT_ENV, "./rel/./cfg");
        assert_eq!(s(&locate(None, &env).unwrap().root), "/work/dir/rel/cfg");
    }

    #[test]
    fn absolute_roots_of_either_convention_are_not_rewritten() {
        // A POSIX root on Windows must not gain the current drive, and a
        // Windows root keeps its drive, with forward slashes, as abs_root does.
        assert_eq!(abs_root("/data/.m3", &base()).unwrap(), "/data/.m3");
        assert_eq!(
            abs_root(r"C:\Users\x\.m3", &base()).unwrap(),
            "C:/Users/x/.m3"
        );
        assert_eq!(
            abs_root(r"\\srv\share\m3", &base()).unwrap(),
            "//srv/share/m3"
        );
    }

    #[test]
    fn no_home_and_no_variable_is_an_error_naming_the_fix() {
        let e = locate(None, &MapEnv::new()).unwrap_err();
        assert!(
            e.contains("--config") && e.contains("M3_CONFIG_ROOT"),
            "{e}"
        );
    }

    #[test]
    fn tier_names_match_m3() {
        assert_eq!(Tier::Arg.as_str(), "arg:--config");
        assert_eq!(Tier::Env.as_str(), "env:M3_CONFIG_ROOT");
        assert_eq!(Tier::Legacy.as_str(), "legacy:M3_MEMORY_ROOT/config");
        assert_eq!(Tier::Default.as_str(), "default:~/.m3/config");
    }
}
