//! Reading settings from m3_config.toml, `--set` and legacy environment names.
//!
//! Precedence for one key, as m3's registry has it
//! (`bin/m3_core/config/resolve.py`): `--set` > m3_config.toml > the key's
//! legacy environment name > the default. A file value is checked against the
//! key's TOML type (`Key.coerce_file_value`); an environment value is parsed
//! with m3's parser for the kind (`registry.parse_bool` / `parse_int`).
//! An invalid value is an error naming the key, where it was set and the fix,
//! never a silent fall back to the default (DESIGN_PHILOSOPHIES §3).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::env::{Env, ProcessEnv};
use crate::roots::{expand_user, Location};

/// The flag that carries settings to a child, spelled as m3 spells it.
pub const SET_FLAG: &str = "--set";

/// What a key holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Bool,
    Int { min: i64, max: i64 },
    Str,
    Path,
}

/// One setting: its m3 registry name (`table.leaf`), kind, and the legacy
/// environment names honoured below the file.
#[derive(Debug, Clone, Copy)]
pub struct Key {
    pub name: &'static str,
    pub kind: Kind,
    pub legacy_env: &'static [&'static str],
    /// Whether `--set` may change it. A locator (where files are, e.g.
    /// `secrets.file`) is not a flag in m3 either: use `--config`.
    pub flag: bool,
}

impl Key {
    pub const fn bool(name: &'static str, legacy_env: &'static [&'static str]) -> Key {
        Key {
            name,
            kind: Kind::Bool,
            legacy_env,
            flag: true,
        }
    }
    pub const fn int(
        name: &'static str,
        legacy_env: &'static [&'static str],
        min: i64,
        max: i64,
    ) -> Key {
        Key {
            name,
            kind: Kind::Int { min, max },
            legacy_env,
            flag: true,
        }
    }
    pub const fn string(name: &'static str, legacy_env: &'static [&'static str]) -> Key {
        Key {
            name,
            kind: Kind::Str,
            legacy_env,
            flag: true,
        }
    }
    pub const fn path(name: &'static str, legacy_env: &'static [&'static str]) -> Key {
        Key {
            name,
            kind: Kind::Path,
            legacy_env,
            flag: true,
        }
    }
    /// The same key, refused by `--set`.
    pub const fn locator(self) -> Key {
        Key {
            flag: false,
            ..self
        }
    }

    /// `[table].leaf`, as m3's messages show a key (`loader.show_key`).
    pub fn shown(&self) -> String {
        match self.name.rsplit_once('.') {
            Some((table, leaf)) => format!("[{table}].{leaf}"),
            None => self.name.to_string(),
        }
    }

    /// `[table]\nleaf = <value>`: how to write it in the file.
    pub fn file_line(&self, value: &str) -> String {
        match self.name.rsplit_once('.') {
            Some((table, leaf)) => format!("[{table}] {leaf} = {value}"),
            None => format!("{} = {value}", self.name),
        }
    }
}

/// A resolved value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Str(String),
    Path(PathBuf),
}

/// Where a value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Set { key: String },
    File { key: String, file: PathBuf },
    LegacyEnv { name: String, key: String },
    Default,
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Set { key } => write!(f, "{SET_FLAG} {key}"),
            Source::File { key, file } => {
                let shown = match key.rsplit_once('.') {
                    Some((t, l)) => format!("[{t}].{l}"),
                    None => key.clone(),
                };
                write!(f, "{shown} in {}", file.display())
            }
            Source::LegacyEnv { name, key } => {
                write!(f, "the environment variable {name} (legacy name of {key})")
            }
            Source::Default => f.write_str("the default"),
        }
    }
}

/// A refused configuration: the message names the file or flag, the key, what
/// was found and what to change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

/// m3's boolean spelling (`registry.parse_bool`).
pub fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Say `msg` once per `tag` for the life of the process.
fn once(tag: &str) -> bool {
    static SEEN: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    let mut seen = SEEN.lock().unwrap_or_else(|p| p.into_inner());
    seen.insert(tag.to_string())
}

fn toml_type(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "a string",
        toml::Value::Integer(_) => "an integer",
        toml::Value::Float(_) => "a float",
        toml::Value::Boolean(_) => "a boolean",
        toml::Value::Datetime(_) => "a datetime",
        toml::Value::Array(_) => "an array",
        toml::Value::Table(_) => "a table",
    }
}

fn expected(kind: Kind) -> String {
    match kind {
        Kind::Bool => "a boolean (true or false)".into(),
        Kind::Int { min, max } => format!("an integer from {min} to {max}"),
        Kind::Str => "a string".into(),
        Kind::Path => "a path (a non-empty string)".into(),
    }
}

fn int_in_range(kind: Kind, v: i64) -> Result<Value, String> {
    match kind {
        Kind::Int { min, max } if v < min || v > max => {
            Err(format!("{v} is out of range: expected {}", expected(kind)))
        }
        _ => Ok(Value::Int(v)),
    }
}

fn path_value(raw: &str, home: Option<&dyn Env>) -> Result<Value, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("an empty string is not a path".into());
    }
    let expanded = match home {
        Some(env) => expand_user(raw, env),
        None => raw.to_string(),
    };
    // Rebuilt from its components, as pathlib does: one separator style on
    // Windows (`private/s.toml` -> `private\s.toml`), `.` parts dropped.
    Ok(Value::Path(Path::new(&expanded).components().collect()))
}

/// A TOML value (file or `--set`) checked against the key's TOML type.
fn from_toml(key: &Key, v: &toml::Value, home: Option<&dyn Env>) -> Result<Value, String> {
    match (key.kind, v) {
        (Kind::Bool, toml::Value::Boolean(b)) => Ok(Value::Bool(*b)),
        (Kind::Int { .. }, toml::Value::Integer(i)) => int_in_range(key.kind, *i),
        (Kind::Str, toml::Value::String(s)) => Ok(Value::Str(s.clone())),
        (Kind::Path, toml::Value::String(s)) => path_value(s, home),
        (kind, other) => Err(format!(
            "expected {}, found {} {other}",
            expected(kind),
            toml_type(other)
        )),
    }
}

/// An environment value parsed by the kind's parser.
fn from_text(key: &Key, raw: &str, home: Option<&dyn Env>) -> Result<Value, String> {
    match key.kind {
        Kind::Bool => parse_bool(raw).map(Value::Bool).ok_or_else(|| {
            format!("{raw:?} is not a boolean; use one of true/false, 1/0, yes/no, on/off")
        }),
        Kind::Int { .. } => match raw.trim().parse::<i64>() {
            Ok(v) => int_in_range(key.kind, v),
            Err(_) => Err(format!("{raw:?} is not an integer")),
        },
        Kind::Str => Ok(Value::Str(raw.to_string())),
        Kind::Path => path_value(raw, home),
    }
}

/// Dotted names of the non-table values (`loader._leaves`).
fn leaves(table: &toml::Table, prefix: &str, out: &mut BTreeMap<String, toml::Value>) {
    for (k, v) in table {
        let name = format!("{prefix}{k}");
        match v {
            toml::Value::Table(t) => leaves(t, &format!("{name}."), out),
            other => {
                out.insert(name, other.clone());
            }
        }
    }
}

/// The settings a `--set` carries, already checked against the known keys.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides(BTreeMap<String, toml::Value>);

impl Overrides {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }
}

/// m3's `^...^` strings (`settings._carets_to_toml`): they survive double
/// quotes in bash, zsh, PowerShell and cmd.exe alike. `^^` inside one is a
/// literal `^`; a `^` inside a TOML `"..."` / `'...'` string is left alone.
fn carets_to_toml(text: &str) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let (mut out, mut i) = (String::new(), 0usize);
    while i < chars.len() {
        let c = chars[i];
        if c == '^' {
            let start = i;
            i += 1;
            let mut buf = String::new();
            loop {
                if i >= chars.len() {
                    return Err(format!(
                        "the ^ string at character {} is not closed; end it with ^ (^^ is a literal ^)",
                        start + 1
                    ));
                }
                if chars[i] == '^' {
                    if chars.get(i + 1) == Some(&'^') {
                        buf.push('^');
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                buf.push(chars[i]);
                i += 1;
            }
            out.push('"');
            for ch in buf.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c if (c as u32) < 0x20 || c == '\u{7f}' => {
                        out.push_str(&format!("\\u{:04x}", c as u32))
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
        } else if c == '"' || c == '\'' {
            let mut j = i + 1;
            while j < chars.len() && chars[j] != c {
                j += if c == '"' && chars[j] == '\\' { 2 } else { 1 };
            }
            let end = (j + 1).min(chars.len());
            out.extend(&chars[i..end]);
            i = j + 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

fn set_refused(problems: &[String]) -> ConfigError {
    let mut msg =
        format!("observed: {SET_FLAG} refused; nothing was applied and the command did not run:");
    for p in problems {
        msg.push_str("\n  - ");
        msg.push_str(p);
    }
    msg.push_str(&format!(
        "\ninspect: the format is \"{SET_FLAG}={{key=value, ...}}\" (one TOML table in double quotes; \
         strings in ^...^, numbers and true/false bare), with m3_config.toml's key names"
    ));
    ConfigError(msg)
}

/// The `{...}` of `--set={...}` (m3 `settings.parse_set`): a TOML inline
/// table with m3_config.toml's key names. Every key must be in `known` and
/// settable, every value of the key's type; all problems are reported at once
/// and nothing applies unless all are valid.
pub fn parse_set(text: &str, known: &[Key]) -> Result<Overrides, ConfigError> {
    let text = text.trim();
    if !(text.starts_with('{') && text.ends_with('}')) {
        // Never echo the token whole: it could carry a secret.
        let head: String = text
            .split('=')
            .next()
            .unwrap_or("")
            .chars()
            .take(40)
            .collect();
        let shown = if text.contains('=') {
            format!("{head}=...")
        } else {
            head
        };
        return Err(set_refused(&[format!(
            "got {SET_FLAG}={shown}, not a {{...}} table. If you typed braces, your shell may have \
             split them: quote the whole token, e.g. \"{SET_FLAG}={{embed.server.port=8083}}\""
        )]));
    }
    let translated = carets_to_toml(text).map_err(|e| set_refused(&[e]))?;
    let doc: toml::Table = toml::from_str(&format!("set = {translated}")).map_err(|e| {
        // The parser's position is in the translated text: keep the reason only.
        let reason = e.message().to_string();
        set_refused(&[format!("the table does not parse ({reason})")])
    })?;
    let mut flat = BTreeMap::new();
    if let Some(toml::Value::Table(t)) = doc.get("set") {
        leaves(t, "", &mut flat);
    }
    let mut problems = Vec::new();
    for (name, raw) in &flat {
        match known.iter().find(|k| k.name == name) {
            None => {
                let env_hit = known.iter().find(|k| k.legacy_env.contains(&name.as_str()));
                problems.push(match env_hit {
                    Some(k) => format!("{name} is an environment-variable name; the setting is {}", k.name),
                    None => format!(
                        "{name} is not a setting this program reads; it reads: {}",
                        known.iter().filter(|k| k.flag).map(|k| k.name).collect::<Vec<_>>().join(", ")
                    ),
                });
            }
            Some(k) if !k.flag => problems.push(format!(
                "{name} decides which files are read, so it is not a flag: set it in m3_config.toml \
                 ({}) and pass --config <file>",
                k.file_line("...")
            )),
            Some(k) => {
                if let Err(e) = from_toml(k, raw, None) {
                    problems.push(format!("{name} = {raw} is not valid: {e}"));
                }
            }
        }
    }
    if problems.is_empty() {
        Ok(Overrides(flat))
    } else {
        Err(set_refused(&problems))
    }
}

/// m3_config.toml as read, plus the `--set` values and the environment the
/// legacy names are read from.
pub struct Config {
    location: Location,
    present: bool,
    file: BTreeMap<String, toml::Value>,
    set: Overrides,
    aliases: Option<Box<dyn Env>>,
    home: Option<Box<dyn Env>>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("location", &self.location)
            .field("present", &self.present)
            .field("file", &self.file)
            .field("set", &self.set)
            .field("aliases", &self.aliases.is_some())
            .finish()
    }
}

impl Config {
    /// Read the file at `location`. An absent file is not an error (every key
    /// takes its default); an unreadable or invalid one is refused with the
    /// path and the fix. `aliases`: where legacy environment names are read
    /// (`None`: not at all, e.g. for what a supervised service will see).
    /// `home` expands `~` in path values.
    pub fn load(
        location: Location,
        set: Overrides,
        aliases: Option<Box<dyn Env>>,
        home: Option<Box<dyn Env>>,
    ) -> Result<Config, ConfigError> {
        let path = &location.file;
        let (present, file) = match std::fs::read(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (false, BTreeMap::new()),
            Err(e) => {
                return Err(ConfigError(format!(
                    "cannot read the m3 config file {} ({e}; located by {}). Fix its permissions, \
                     or point at another file with --config <file>",
                    path.display(),
                    location.tier
                )))
            }
            Ok(bytes) => {
                let text = String::from_utf8(bytes).map_err(|e| {
                    ConfigError(format!(
                        "the m3 config file {} is not UTF-8 ({e}); save it as UTF-8",
                        path.display()
                    ))
                })?;
                let doc: toml::Table = toml::from_str(&text).map_err(|e| {
                    ConfigError(format!(
                        "the m3 config file {} is not valid TOML: {e}inspect: `m3 config show --origin`; \
                         fix the line named, then retry",
                        path.display()
                    ))
                })?;
                let mut flat = BTreeMap::new();
                leaves(&doc, "", &mut flat);
                (true, flat)
            }
        };
        Ok(Config {
            location,
            present,
            file,
            set,
            aliases,
            home,
        })
    }

    pub fn location(&self) -> &Location {
        &self.location
    }

    /// Whether m3_config.toml existed when it was read.
    pub fn file_present(&self) -> bool {
        self.present
    }

    fn home(&self) -> Option<&dyn Env> {
        self.home.as_deref()
    }

    /// The value of `key` and where it came from, or `None` when nothing sets it.
    pub fn get(&self, key: &Key) -> Result<Option<(Value, Source)>, ConfigError> {
        if let Some(raw) = self.set.0.get(key.name) {
            let v = from_toml(key, raw, self.home())
                .map_err(|e| ConfigError(format!("{SET_FLAG} {}: {e}", key.name)))?;
            return Ok(Some((
                v,
                Source::Set {
                    key: key.name.into(),
                },
            )));
        }
        if let Some(raw) = self.file.get(key.name) {
            let v = from_toml(key, raw, self.home()).map_err(|e| {
                ConfigError(format!(
                    "{}: {}: {e}. Correct it with `m3 config set {} <value>`, or edit the file",
                    self.location.file.display(),
                    key.shown(),
                    key.name
                ))
            })?;
            self.note_shadowed(key);
            return Ok(Some((
                v,
                Source::File {
                    key: key.name.into(),
                    file: self.location.file.clone(),
                },
            )));
        }
        let Some(env) = self.aliases.as_deref() else {
            return Ok(None);
        };
        for name in key.legacy_env {
            let Some(raw) = env.var(name) else { continue };
            let v = from_text(key, &raw, self.home()).map_err(|e| {
                ConfigError(format!(
                    "the environment variable {name} (legacy name of {}) is invalid: {e}. Unset it and \
                     set {} in {}",
                    key.name,
                    key.file_line("<value>"),
                    self.location.file.display()
                ))
            })?;
            if once(&format!("legacy:{name}")) {
                log::warn!(
                    "{name} decides {} ({raw:?}): environment names for m3 settings are deprecated and \
                     rank below m3_config.toml. Set it in {} as {} (`m3 config set {} <value>`), \
                     then unset {name}",
                    key.name,
                    self.location.file.display(),
                    key.file_line("<value>"),
                    key.name
                );
            }
            return Ok(Some((
                v,
                Source::LegacyEnv {
                    name: (*name).into(),
                    key: key.name.into(),
                },
            )));
        }
        Ok(None)
    }

    /// An environment name set but outranked: reported once, never silently
    /// dropped (m3 reports `shadowed` the same way).
    fn note_shadowed(&self, key: &Key) {
        let Some(env) = self.aliases.as_deref() else {
            return;
        };
        for name in key.legacy_env {
            if env.var(name).is_some() && once(&format!("shadowed:{name}")) {
                log::info!(
                    "{name} is set, but {} in {} outranks it; the environment value is not used",
                    key.shown(),
                    self.location.file.display()
                );
            }
        }
    }

    pub fn bool_or(&self, key: &Key, default: bool) -> Result<(bool, Source), ConfigError> {
        match self.get(key)? {
            Some((Value::Bool(b), s)) => Ok((b, s)),
            Some((other, _)) => Err(kind_mismatch(key, &other)),
            None => Ok((default, Source::Default)),
        }
    }

    pub fn int_or<T: TryFrom<i64> + Copy>(
        &self,
        key: &Key,
        default: T,
    ) -> Result<(T, Source), ConfigError> {
        match self.get(key)? {
            Some((Value::Int(i), s)) => T::try_from(i)
                .map(|v| (v, s.clone()))
                .map_err(|_| ConfigError(format!("{}: {i} (from {s}) does not fit", key.name))),
            Some((other, _)) => Err(kind_mismatch(key, &other)),
            None => Ok((default, Source::Default)),
        }
    }

    pub fn string(&self, key: &Key) -> Result<Option<(String, Source)>, ConfigError> {
        match self.get(key)? {
            Some((Value::Str(v), s)) => Ok(Some((v, s))),
            Some((other, _)) => Err(kind_mismatch(key, &other)),
            None => Ok(None),
        }
    }

    pub fn path(&self, key: &Key) -> Result<Option<(PathBuf, Source)>, ConfigError> {
        match self.get(key)? {
            Some((Value::Path(v), s)) => Ok(Some((v, s))),
            Some((other, _)) => Err(kind_mismatch(key, &other)),
            None => Ok(None),
        }
    }

    /// A path relative to the config root (as m3 resolves `secrets.file`).
    pub fn under_root(&self, p: &Path) -> PathBuf {
        if p.is_absolute() || p.has_root() {
            p.to_path_buf()
        } else {
            self.location.root.join(p)
        }
    }
}

fn kind_mismatch(key: &Key, got: &Value) -> ConfigError {
    ConfigError(format!(
        "internal: {} is declared {:?} but resolved to {got:?}",
        key.name, key.kind
    ))
}

/// For callers that take a setting as a PARAMETER and still honour its old
/// environment name for one release (m3-core-py): the value of `name` when set,
/// with one warning naming the parameter and the m3_config.toml key that
/// replace it.
pub fn deprecated_env(name: &str, param: &str, key: &str) -> Option<String> {
    let raw = ProcessEnv.var(name)?;
    if once(&format!("deprecated:{name}")) {
        log::warn!(
            "{name}={raw:?} is used because the caller did not pass {param}=; the environment name is \
             deprecated and stops being read in the next release. The setting is {key} in \
             m3_config.toml, which m3 passes as {param}=: set it there (`m3 config set {key} <value>`) \
             and unset {name}"
        );
    }
    Some(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::roots::Tier;

    const FLAG: Key = Key::bool("crypto.fips_mode", &["M3_FIPS_MODE"]);
    const PORT: Key = Key::int("embed.server.port", &["M3_EMBED_SERVER_PORT"], 1, 65535);
    const GGUF: Key = Key::path("embed.gguf", &["M3_EMBED_GGUF"]);
    const LOC: Key = Key::path("secrets.file", &[]).locator();
    const KNOWN: [Key; 4] = [FLAG, PORT, GGUF, LOC];

    fn config(text: Option<&str>, set: Overrides, env: MapEnv) -> (tempfile::TempDir, Config) {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("m3_config.toml");
        if let Some(t) = text {
            std::fs::write(&file, t).unwrap();
        }
        let loc = Location {
            file,
            root: dir.path().to_path_buf(),
            tier: Tier::Arg,
        };
        let cfg = Config::load(loc, set, Some(Box::new(env)), None).unwrap();
        (dir, cfg)
    }

    #[test]
    fn the_file_outranks_the_legacy_env_name_and_says_so() {
        let env = MapEnv::new()
            .with("M3_FIPS_MODE", "0")
            .with("M3_EMBED_SERVER_PORT", "9999");
        let (_d, cfg) = config(
            Some("[crypto]\nfips_mode = true\n[embed.server]\nport = 8083\n"),
            Overrides::default(),
            env,
        );
        let (on, from) = cfg.bool_or(&FLAG, false).unwrap();
        assert!(on);
        assert!(
            matches!(&from, Source::File { key, .. } if key == "crypto.fips_mode"),
            "{from}"
        );
        assert!(
            from.to_string().starts_with("[crypto].fips_mode in "),
            "{from}"
        );
        let (port, from) = cfg.int_or::<u16>(&PORT, 8082).unwrap();
        assert_eq!(port, 8083);
        assert!(
            from.to_string().starts_with("[embed.server].port in "),
            "{from}"
        );
    }

    #[test]
    fn the_legacy_env_name_applies_only_below_the_file_and_is_reported() {
        let env = MapEnv::new().with("M3_FIPS_MODE", "yes");
        let (_d, cfg) = config(None, Overrides::default(), env);
        let (on, from) = cfg.bool_or(&FLAG, false).unwrap();
        assert!(on);
        assert_eq!(
            from,
            Source::LegacyEnv {
                name: "M3_FIPS_MODE".into(),
                key: "crypto.fips_mode".into()
            }
        );
        let (port, from) = cfg.int_or::<u16>(&PORT, 8082).unwrap();
        assert_eq!((port, from), (8082, Source::Default));
    }

    #[test]
    fn set_outranks_the_file() {
        let set = parse_set("{embed.server.port = 8090, crypto.fips_mode=false}", &KNOWN).unwrap();
        let (_d, cfg) = config(Some("[embed.server]\nport = 8083\n"), set, MapEnv::new());
        assert_eq!(
            cfg.int_or::<u16>(&PORT, 8082).unwrap(),
            (
                8090,
                Source::Set {
                    key: "embed.server.port".into()
                }
            )
        );
    }

    #[test]
    fn a_wrong_type_in_the_file_is_refused_with_the_key_and_file() {
        let (_d, cfg) = config(
            Some("[embed.server]\nport = \"8083\"\n"),
            Overrides::default(),
            MapEnv::new(),
        );
        let e = cfg.int_or::<u16>(&PORT, 8082).unwrap_err().0;
        assert!(
            e.contains("[embed.server].port") && e.contains("m3_config.toml"),
            "{e}"
        );
        assert!(e.contains("expected an integer"), "{e}");
    }

    #[test]
    fn an_invalid_legacy_env_value_is_an_error_not_the_default() {
        let (_d, cfg) = config(
            None,
            Overrides::default(),
            MapEnv::new().with("M3_FIPS_MODE", "maybe"),
        );
        let e = cfg.bool_or(&FLAG, false).unwrap_err().0;
        assert!(
            e.contains("M3_FIPS_MODE") && e.contains("not a boolean"),
            "{e}"
        );
    }

    #[test]
    fn invalid_toml_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("m3_config.toml");
        std::fs::write(&file, "[crypto\nfips_mode = true\n").unwrap();
        let loc = Location {
            file,
            root: dir.path().into(),
            tier: Tier::Arg,
        };
        let e = Config::load(loc, Overrides::default(), None, None)
            .unwrap_err()
            .0;
        assert!(
            e.contains("not valid TOML") && e.contains("m3_config.toml"),
            "{e}"
        );
    }

    #[test]
    fn set_accepts_m3s_caret_strings_and_refuses_bad_entries_all_at_once() {
        let o = parse_set("{embed.gguf=^C:\\m\\a^^b.gguf^}", &KNOWN).unwrap();
        assert_eq!(
            o.0.get("embed.gguf"),
            Some(&toml::Value::String("C:\\m\\a^b.gguf".into()))
        );
        let e = parse_set(
            "{embed.server.port=^x^, M3_FIPS_MODE=true, nope.key=1, secrets.file=^/s^}",
            &KNOWN,
        )
        .unwrap_err()
        .0;
        assert!(e.contains("embed.server.port = \"x\" is not valid"), "{e}");
        assert!(
            e.contains("M3_FIPS_MODE is an environment-variable name"),
            "{e}"
        );
        assert!(e.contains("nope.key is not a setting"), "{e}");
        assert!(e.contains("secrets.file decides which files"), "{e}");
        assert!(parse_set("embed.server.port=1", &KNOWN)
            .unwrap_err()
            .0
            .contains("not a {...} table"));
    }

    #[test]
    fn bool_spelling_is_m3s() {
        for t in ["1", "true", "YES", " on "] {
            assert_eq!(parse_bool(t), Some(true), "{t}");
        }
        for f in ["0", "false", "No", "off"] {
            assert_eq!(parse_bool(f), Some(false), "{f}");
        }
        assert_eq!(parse_bool("2"), None);
    }
}
