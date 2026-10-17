//! Content hashing: SHA-256 keys for dedup and change detection (Phase 1.3 /
//! 3d §4c.2).
//!
//! These digests are not a security use, so they sit OUTSIDE m3's FIPS 140-3
//! boundary (m3-memory docs/FIPS_MODULE_BOUNDARY.md, "Non-security hashes"),
//! like Python's `hashlib.sha256(..., usedforsecurity=False)`. Neither provider
//! here is a validated module: `ring` (the default) and `sha2` (behind the
//! historically named `non-fips-perf` feature, dev-only) give identical digests.

/// Hashing provider abstraction. `RingHasher` is the default;
/// `RustCryptoHasher` is feature-gated behind `non-fips-perf`.
pub trait Hasher {
    fn sha256(&self, data: &[u8]) -> [u8; 32];
}

/// SHA-256 backed by `ring::digest` (not a validated module; see the crate docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct RingHasher;

impl Hasher for RingHasher {
    fn sha256(&self, data: &[u8]) -> [u8; 32] {
        let d = ring::digest::digest(&ring::digest::SHA256, data);
        let mut out = [0u8; 32];
        out.copy_from_slice(d.as_ref());
        out
    }
}

/// SHA-256 backed by the `sha2` crate (hardware SHA-NI). Dev-only.
#[cfg(feature = "non-fips-perf")]
#[derive(Debug, Default, Clone, Copy)]
pub struct RustCryptoHasher;

#[cfg(feature = "non-fips-perf")]
impl Hasher for RustCryptoHasher {
    fn sha256(&self, data: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(data);
        h.finalize().into()
    }
}

/// Identifies the active provider; surfaced in `m3:health` for drift detection.
pub fn active_provider() -> &'static str {
    if cfg!(feature = "non-fips-perf") {
        "sha2 (non-fips-perf)"
    } else {
        "ring"
    }
}

/// Lowercase hex digest — byte-identical to Python `hashlib.sha256().hexdigest()`.
pub fn hex(digest: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Convenience: SHA-256 hex of `data` via the default-feature active provider.
pub fn sha256_hex(data: &[u8]) -> String {
    #[cfg(feature = "non-fips-perf")]
    let digest = RustCryptoHasher.sha256(data);
    #[cfg(not(feature = "non-fips-perf"))]
    let digest = RingHasher.sha256(data);
    log::debug!("m3-hash: computed sha256 via {}", active_provider());
    hex(&digest)
}
