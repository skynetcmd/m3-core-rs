//! FIPS-preserving content hashing (Phase 1.3 / 3d §4c.2).
//!
//! The default build links only a FIPS-validatable provider (`ring`). A
//! non-FIPS fast path (`sha2`) is gated behind the `non-fips-perf` feature and
//! never enabled in production wheels.

/// Hashing provider abstraction. `RingHasher` is the default (FIPS) impl;
/// `RustCryptoHasher` is feature-gated behind `non-fips-perf`.
pub trait Hasher {
    fn sha256(&self, data: &[u8]) -> [u8; 32];
}

/// FIPS-validatable SHA-256 backed by `ring::digest`.
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

/// Non-FIPS SHA-256 backed by the `sha2` crate (hardware SHA-NI). Dev-only.
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
