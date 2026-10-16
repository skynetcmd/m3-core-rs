//! Filesystem-watcher hot path — the mechanical, CPU/syscall-bound parts of
//! the Python `bin/files_memory/walker.py` walk and the staleness hash checks,
//! oxidized for speed.
//!
//! Scope is deliberately narrow. We oxidize the two operations that dominate
//! wall-clock on large trees:
//!   1. the recursive directory sweep (read_dir + stat), and
//!   2. batch content hashing (read file + SHA-256).
//!
//! The nuanced, config-coupled filters in the Python walker — gitignore
//! semantics, binary sniffing, filetype detection, glob include/exclude — stay
//! in Python. They are cheap relative to I/O and risky to port without
//! behavioral drift, so the Python orchestrator keeps deciding *what* to ingest;
//! this crate just makes the *gathering* fast. The walk returns every
//! non-ignored entry with `(path, size, mtime, is_dir)`; Python applies the
//! rest of its filter pipeline to that list.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

/// One filesystem entry produced by the sweep. Mirrors the fields the Python
/// walker needs cheaply up front; the heavier filetype/sha fields are filled in
/// later by Python / `hash_files`.
#[derive(Debug, Clone)]
pub struct Entry {
    pub path: String,
    pub size: u64,
    /// Modification time as seconds since the Unix epoch (f64 to match Python's
    /// `st_mtime`). 0.0 when the platform reports no mtime or one before the
    /// epoch.
    pub mtime: f64,
    pub is_dir: bool,
}

/// An entry the sweep could not read (or would not descend), with the reason.
/// Accumulated, never fatal — parity with the Python walker.
#[derive(Debug, Clone)]
pub struct WalkError {
    pub path: String,
    pub error: String,
}

/// Everything a sweep found, plus every entry it had to skip and why.
#[derive(Debug, Clone, Default)]
pub struct Walk {
    pub entries: Vec<Entry>,
    pub errors: Vec<WalkError>,
}

/// Recursion ceiling. With `follow_symlinks` a link back up the tree would
/// otherwise recurse until the stack overflowed — which aborts the whole
/// Python process. Directory trees this deep are not real source trees.
pub const MAX_WALK_DEPTH: usize = 256;

/// Recursively sweep `root`, skipping any directory whose basename is in
/// `dir_ignores` (the cheap first-stage filter the Python walker applies before
/// the per-directory gitignore matcher). Symlinked directories are descended
/// only when `follow_symlinks` is true, each real directory at most once (a
/// link cycle is reported, not followed). `max_depth` is measured from `root`
/// (`Some(0)` = root's direct children only); `None` = unbounded up to
/// [`MAX_WALK_DEPTH`].
///
/// An unreadable or missing ROOT is an error: an empty result would read as
/// "an empty directory". Errors below the root (permission denied, races,
/// cycles, the depth ceiling) are collected in `Walk::errors` and the sweep
/// continues — parity with the Python walker, which accumulates errors.
/// Directories are included in the output (so Python can still see them) but
/// the recursion itself honors the ignore set.
pub fn walk(
    root: &str,
    dir_ignores: &[String],
    max_depth: Option<usize>,
    follow_symlinks: bool,
) -> m3_error::Result<Walk> {
    let root_path = Path::new(root);
    let rd = std::fs::read_dir(root_path).map_err(|e| {
        m3_error::M3Error::Io(std::io::Error::new(
            e.kind(),
            format!("cannot read walk root '{root}': {e}"),
        ))
    })?;
    let mut w = Walk::default();
    let ignore: HashSet<&str> = dir_ignores.iter().map(|s| s.as_str()).collect();
    let mut visited = HashSet::new();
    if let Ok(c) = std::fs::canonicalize(root_path) {
        visited.insert(c);
    }
    let ctx = Ctx {
        max_depth,
        follow_symlinks,
        ignore: &ignore,
    };
    recurse(rd, 0, &ctx, &mut visited, &mut w);
    Ok(w)
}

struct Ctx<'a> {
    max_depth: Option<usize>,
    follow_symlinks: bool,
    ignore: &'a HashSet<&'a str>,
}

fn record(w: &mut Walk, path: &Path, error: impl std::fmt::Display) {
    w.errors.push(WalkError {
        path: path.to_string_lossy().into_owned(),
        error: error.to_string(),
    });
}

fn recurse(
    rd: std::fs::ReadDir,
    depth: usize,
    ctx: &Ctx<'_>,
    visited: &mut HashSet<PathBuf>,
    w: &mut Walk,
) {
    for entry in rd {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                record(w, Path::new("(directory listing)"), e);
                continue;
            }
        };
        let path = entry.path();
        // Use symlink_metadata first so we can apply the symlink policy without
        // following; only follow into the target when allowed.
        let lmeta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                record(w, &path, e);
                continue;
            }
        };
        let is_symlink = lmeta.file_type().is_symlink();
        if is_symlink && !ctx.follow_symlinks {
            continue;
        }
        // Resolve the effective metadata (follow when permitted).
        let meta = if is_symlink {
            match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    record(w, &path, format!("broken symlink: {e}"));
                    continue;
                }
            }
        } else {
            lmeta
        };

        let is_dir = meta.is_dir();
        let path_str = path.to_string_lossy().to_string();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        if !is_dir {
            w.entries.push(Entry {
                path: path_str,
                size: meta.len(),
                mtime,
                is_dir: false,
            });
            continue;
        }
        let basename = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        // Cheap dir-ignore stage (parity with `bn in dir_ignores`).
        if ctx.ignore.contains(basename.as_str()) {
            continue;
        }
        w.entries.push(Entry {
            path: path_str,
            size: 0,
            mtime,
            is_dir: true,
        });
        let child_depth = depth + 1;
        if ctx.max_depth.is_some_and(|md| child_depth > md) {
            continue;
        }
        if child_depth > MAX_WALK_DEPTH {
            record(
                w,
                &path,
                format!("not descended: deeper than the {MAX_WALK_DEPTH}-level walk ceiling"),
            );
            continue;
        }
        if ctx.follow_symlinks {
            match std::fs::canonicalize(&path) {
                Ok(real) => {
                    if !visited.insert(real.clone()) {
                        record(
                            w,
                            &path,
                            format!(
                                "not descended: {} was already walked (symlink cycle or alias)",
                                real.display()
                            ),
                        );
                        continue;
                    }
                }
                Err(e) => {
                    record(w, &path, format!("cannot resolve: {e}"));
                    continue;
                }
            }
        }
        match std::fs::read_dir(&path) {
            Ok(rd) => recurse(rd, child_depth, ctx, visited, w),
            Err(e) => record(w, &path, e),
        }
    }
}

/// Batch-hash file contents in parallel. Returns `(path, result)` pairs in the
/// SAME order as the input. `result` is `Ok(hex_sha256)` or `Err(message)` for
/// files that could not be read — the caller decides how to treat failures
/// (parity with the Python staleness path, which records an error and
/// continues rather than aborting the batch).
///
/// The digest is byte-identical to Python's `file_content_sha256`
/// (streaming SHA-256, hex) because both go through the same SHA-256 over the
/// raw file bytes; here via `m3_hash::sha256_hex` (a content key, outside the FIPS boundary).
pub fn hash_files(paths: &[String]) -> Vec<(String, Result<String, String>)> {
    paths
        .par_iter()
        .map(|p| {
            let res = std::fs::read(p)
                .map(|bytes| m3_hash::sha256_hex(&bytes))
                .map_err(|e| format!("{e}"));
            (p.clone(), res)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_known_sha256() {
        // SHA-256 of "hello" — must match Python hashlib.sha256(b"hello").
        let dir = std::env::temp_dir().join("m3_ingest_hash_test");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("hello.txt");
        std::fs::write(&f, b"hello").unwrap();
        let paths = vec![f.to_string_lossy().to_string()];
        let out = hash_files(&paths);
        assert_eq!(
            out[0].1.as_ref().unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_missing_file_is_err_not_panic() {
        let out = hash_files(&["/nonexistent/path/xyz".to_string()]);
        assert!(out[0].1.is_err());
    }

    #[test]
    fn walk_finds_files_and_honors_dir_ignore() {
        let dir = std::env::temp_dir().join("m3_ingest_walk_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("keep")).unwrap();
        std::fs::create_dir_all(dir.join("node_modules")).unwrap();
        std::fs::write(dir.join("keep/a.txt"), b"a").unwrap();
        std::fs::write(dir.join("node_modules/b.txt"), b"b").unwrap();

        let entries = walk(
            &dir.to_string_lossy(),
            &["node_modules".to_string()],
            None,
            false,
        )
        .unwrap()
        .entries;
        let files: Vec<&str> = entries
            .iter()
            .filter(|e| !e.is_dir)
            .map(|e| e.path.as_str())
            .collect();
        assert!(files.iter().any(|p| p.ends_with("a.txt")));
        // node_modules subtree must be pruned.
        assert!(!files.iter().any(|p| p.ends_with("b.txt")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn walk_max_depth_zero_is_direct_children_only() {
        let dir = std::env::temp_dir().join("m3_ingest_depth_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("top.txt"), b"t").unwrap();
        std::fs::write(dir.join("sub/deep.txt"), b"d").unwrap();

        let entries = walk(&dir.to_string_lossy(), &[], Some(0), false)
            .unwrap()
            .entries;
        let files: Vec<&str> = entries
            .iter()
            .filter(|e| !e.is_dir)
            .map(|e| e.path.as_str())
            .collect();
        assert!(files.iter().any(|p| p.ends_with("top.txt")));
        assert!(!files.iter().any(|p| p.ends_with("deep.txt")));
        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn a_missing_root_is_an_error_not_an_empty_tree() {
        let e = walk("/nonexistent/m3/ingest/root", &[], None, false).unwrap_err();
        assert!(e.to_string().contains("cannot read walk root"), "{e}");
        assert!(
            matches!(e, m3_error::M3Error::Io(ref io) if io.kind() == std::io::ErrorKind::NotFound),
            "{e}"
        );
    }

    /// A link back up the tree is walked once and reported, not followed
    /// until the stack overflows (which aborted Python).
    #[cfg(unix)]
    #[test]
    fn a_symlink_cycle_is_reported_not_followed() {
        let dir = std::env::temp_dir().join("m3_ingest_cycle_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::os::unix::fs::symlink(&dir, dir.join("a/loop")).unwrap();
        let w = walk(&dir.to_string_lossy(), &[], None, true).unwrap();
        assert!(
            w.errors.iter().any(|e| e.error.contains("already walked")),
            "{:?}",
            w.errors
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
