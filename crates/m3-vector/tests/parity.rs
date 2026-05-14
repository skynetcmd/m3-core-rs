use m3_vector::{blob_as_f32, cosine, f32_as_blob, mmr_rerank};
use proptest::prelude::*;
use std::collections::HashSet;

fn vec_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-100.0f32..100.0f32, 1..64)
}

fn pair_strategy() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (1usize..64).prop_flat_map(|n| {
        (
            prop::collection::vec(-100.0f32..100.0f32, n),
            prop::collection::vec(-100.0f32..100.0f32, n),
        )
    })
}

proptest! {
    #[test]
    fn cosine_self_is_one(v in vec_strategy()) {
        let norm: f32 = v.iter().map(|x| x * x).sum();
        prop_assume!(norm > 1e-6);
        let c = cosine(&v, &v).unwrap();
        prop_assert!((c - 1.0).abs() < 1e-4, "cosine self = {c}");
    }

    #[test]
    fn cosine_is_symmetric((a, b) in pair_strategy()) {
        let ab = cosine(&a, &b).unwrap();
        let ba = cosine(&b, &a).unwrap();
        prop_assert!((ab - ba).abs() < 1e-5);
    }

    #[test]
    fn cosine_in_range((a, b) in pair_strategy()) {
        let c = cosine(&a, &b).unwrap();
        prop_assert!((-1.0..=1.0).contains(&c), "cosine out of range: {c}");
    }

    #[test]
    fn bytemuck_roundtrip(v in vec_strategy()) {
        let bytes = f32_as_blob(&v);
        let back = blob_as_f32(bytes).unwrap();
        prop_assert_eq!(back, v.as_slice());
    }

    #[test]
    fn mmr_returns_unique_min_k(
        query in vec_strategy(),
        corpus in prop::collection::vec(vec_strategy(), 1..20),
        lambda in 0.0f32..=1.0f32,
        k in 0usize..25,
    ) {
        let dim = query.len();
        let fixed: Vec<Vec<f32>> = corpus.iter()
            .map(|v| {
                let mut v = v.clone();
                v.resize(dim, 0.0);
                v
            })
            .collect();
        let refs: Vec<&[f32]> = fixed.iter().map(|v| v.as_slice()).collect();
        let sel = mmr_rerank(&query, &refs, lambda, k).unwrap();
        prop_assert_eq!(sel.len(), k.min(refs.len()));
        let uniq: HashSet<_> = sel.iter().collect();
        prop_assert_eq!(uniq.len(), sel.len());
        prop_assert!(sel.iter().all(|&i| i < refs.len()));
    }
}
