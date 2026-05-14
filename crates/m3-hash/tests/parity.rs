//! SHA-256 known-answer vectors — parity gate against Python `hashlib`.

use m3_hash::{hex, sha256_hex, Hasher, RingHasher};

#[test]
fn empty_string_vector() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn abc_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn ring_hasher_matches_helper() {
    let d = RingHasher.sha256(b"abc");
    assert_eq!(
        hex(&d),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[cfg(feature = "non-fips-perf")]
#[test]
fn rustcrypto_matches_ring() {
    use m3_hash::RustCryptoHasher;
    for input in [&b""[..], b"abc", b"the quick brown fox"] {
        assert_eq!(RingHasher.sha256(input), RustCryptoHasher.sha256(input));
    }
}
