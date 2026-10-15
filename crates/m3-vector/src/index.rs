//! A resident, int8-quantised vector index for semantic candidate retrieval.
//!
//! m3's SQLite search re-read every candidate's float32 blob from the database
//! on each query: about half of a ~200 ms search on a 21k-vector store, and the
//! term that grows with the store (0.45 s at 50k memories, 1.55 s at 150k,
//! measured 2026-10-15). This index keeps the vectors in memory instead, at a
//! quarter of the float32 size (one i8 per dimension plus one f32 scale per
//! vector), and scans them all per query. The caller re-scores the top few
//! hundred with exact float32 cosine, so quantisation decides only which rows
//! reach the exact pass, never the final score.
//!
//! Rows are keyed by the caller's integer key (m3 uses the embedding row's
//! rowid). Search can be restricted to an allowed key set, so tenancy and scope
//! filtering stays in SQL as an id-only query.

use std::collections::{BinaryHeap, HashMap};

use m3_error::{M3Error, Result};
use rayon::prelude::*;

/// One quantised vector: `codes[i] * scale` ~= the unit-normalised value.
pub struct VectorIndex {
    dim: usize,
    keys: Vec<i64>,
    codes: Vec<i8>,
    scales: Vec<f32>,
    pos: HashMap<i64, usize>,
}

/// A (score, key) pair ordered so a `BinaryHeap` of them is a MIN-heap on score,
/// which keeps the k best seen so far.
#[derive(PartialEq)]
struct Hit(f32, i64);
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.0.total_cmp(&self.0).then_with(|| other.1.cmp(&self.1))
    }
}

impl VectorIndex {
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(M3Error::Config("index dimension must be > 0".into()));
        }
        Ok(Self { dim, keys: Vec::new(), codes: Vec::new(), scales: Vec::new(), pos: HashMap::new() })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Resident bytes held by the vectors (codes and scales).
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }

    pub fn contains(&self, key: i64) -> bool {
        self.pos.contains_key(&key)
    }

    /// Insert or replace `key`'s vector. A zero vector is stored as zeros and
    /// scores 0 against everything.
    pub fn upsert(&mut self, key: i64, vec: &[f32]) -> Result<()> {
        if vec.len() != self.dim {
            return Err(M3Error::VectorDimMismatch { expected: self.dim, got: vec.len() });
        }
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let peak = vec.iter().fold(0f32, |m, x| m.max(x.abs()));
        let (scale, inv) = if norm > 0.0 && peak > 0.0 {
            let unit_peak = peak / norm;
            (unit_peak / 127.0, 127.0 / peak)
        } else {
            (0.0, 0.0)
        };
        let row = |out: &mut [i8]| {
            for (o, x) in out.iter_mut().zip(vec) {
                *o = (x * inv).round().clamp(-127.0, 127.0) as i8;
            }
        };
        match self.pos.get(&key) {
            Some(&i) => {
                row(&mut self.codes[i * self.dim..(i + 1) * self.dim]);
                self.scales[i] = scale;
            }
            None => {
                let i = self.keys.len();
                self.keys.push(key);
                self.codes.resize(self.codes.len() + self.dim, 0);
                row(&mut self.codes[i * self.dim..]);
                self.scales.push(scale);
                self.pos.insert(key, i);
            }
        }
        Ok(())
    }

    /// Remove `key` if present (swap-remove: O(dim)). Returns whether it was there.
    pub fn remove(&mut self, key: i64) -> bool {
        let Some(i) = self.pos.remove(&key) else { return false };
        let last = self.keys.len() - 1;
        if i != last {
            let moved = self.keys[last];
            self.keys[i] = moved;
            self.scales[i] = self.scales[last];
            let (head, tail) = self.codes.split_at_mut(last * self.dim);
            head[i * self.dim..(i + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            self.pos.insert(moved, i);
        }
        self.keys.pop();
        self.scales.pop();
        self.codes.truncate(last * self.dim);
        true
    }

    fn score(&self, q: &[f32], i: usize) -> f32 {
        let row = &self.codes[i * self.dim..(i + 1) * self.dim];
        let dot: f32 = row.iter().zip(q).map(|(c, x)| *c as f32 * x).sum();
        dot * self.scales[i]
    }

    /// The `k` best keys by approximate cosine to `query`, best first. With
    /// `allowed`, only those keys are considered (unknown keys are ignored).
    pub fn search(&self, query: &[f32], k: usize, allowed: Option<&[i64]>) -> Result<Vec<(i64, f32)>> {
        if query.len() != self.dim {
            return Err(M3Error::VectorDimMismatch { expected: self.dim, got: query.len() });
        }
        if k == 0 || self.keys.is_empty() {
            return Ok(Vec::new());
        }
        let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Ok(Vec::new());
        }
        let q: Vec<f32> = query.iter().map(|x| x / norm).collect();
        let rows: Vec<usize> = match allowed {
            Some(keys) => keys.iter().filter_map(|key| self.pos.get(key).copied()).collect(),
            None => (0..self.keys.len()).collect(),
        };
        let top = rows
            .par_chunks(4096)
            .map(|chunk| {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                for &i in chunk {
                    heap.push(Hit(self.score(&q, i), self.keys[i]));
                    if heap.len() > k {
                        heap.pop();
                    }
                }
                heap
            })
            .reduce(BinaryHeap::new, |mut a, b| {
                for h in b {
                    a.push(h);
                    if a.len() > k {
                        a.pop();
                    }
                }
                a
            });
        let mut out: Vec<(i64, f32)> = top.into_iter().map(|Hit(s, key)| (key, s)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vectors (xorshift), no extra dependency.
    fn vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    fn exact_top(q: &[f32], vs: &[Vec<f32>], k: usize) -> Vec<i64> {
        let mut s: Vec<(i64, f32)> = vs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as i64, crate::cosine(q, v).unwrap()))
            .collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1));
        s.into_iter().take(k).map(|x| x.0).collect()
    }

    #[test]
    fn the_exact_top_hits_survive_quantisation() {
        let (n, dim) = (5000, 256);
        let vs = vectors(n, dim, 7);
        let mut idx = VectorIndex::new(dim).unwrap();
        for (i, v) in vs.iter().enumerate() {
            idx.upsert(i as i64, v).unwrap();
        }
        let mut found = 0;
        for q in vectors(20, dim, 99) {
            let truth = exact_top(&q, &vs, 10);
            // The caller re-scores a wider candidate set exactly; 50 here.
            let cand: Vec<i64> = idx.search(&q, 50, None).unwrap().into_iter().map(|x| x.0).collect();
            found += truth.iter().filter(|t| cand.contains(t)).count();
        }
        assert!(found >= 195, "{found}/200 exact top-10 hits reached the top-50 candidates");
    }

    #[test]
    fn scores_track_exact_cosine() {
        let vs = vectors(200, 128, 3);
        let mut idx = VectorIndex::new(128).unwrap();
        for (i, v) in vs.iter().enumerate() {
            idx.upsert(i as i64, v).unwrap();
        }
        let q = &vs[5];
        for (key, approx) in idx.search(q, 200, None).unwrap() {
            let exact = crate::cosine(q, &vs[key as usize]).unwrap();
            assert!((approx - exact).abs() < 0.02, "key {key}: {approx} vs {exact}");
        }
    }

    #[test]
    fn allowed_keys_restrict_the_search() {
        let vs = vectors(100, 32, 11);
        let mut idx = VectorIndex::new(32).unwrap();
        for (i, v) in vs.iter().enumerate() {
            idx.upsert(i as i64, v).unwrap();
        }
        let hits = idx.search(&vs[0], 10, Some(&[3, 4, 5, 999])).unwrap();
        let keys: Vec<i64> = hits.iter().map(|h| h.0).collect();
        assert_eq!(keys.len(), 3);
        assert!(keys.iter().all(|k| [3, 4, 5].contains(k)), "{keys:?}");
    }

    #[test]
    fn upsert_replaces_and_remove_keeps_the_rest_findable() {
        let vs = vectors(50, 16, 5);
        let mut idx = VectorIndex::new(16).unwrap();
        for (i, v) in vs.iter().enumerate() {
            idx.upsert(i as i64, v).unwrap();
        }
        idx.upsert(7, &vs[20]).unwrap();
        assert_eq!(idx.len(), 50, "an upsert of a known key replaces it");
        assert!(idx.remove(0));
        assert!(!idx.remove(0));
        assert_eq!(idx.len(), 49);
        // Every remaining key still finds itself first (the moved row included).
        for key in 1..50i64 {
            let v = if key == 7 { &vs[20] } else { &vs[key as usize] };
            let best = idx.search(v, 1, None).unwrap()[0].0;
            assert!(best == key || (key == 7 && best == 20) || (key == 20 && best == 7), "key {key} found {best}");
        }
    }

    #[test]
    fn bad_dimensions_are_errors() {
        let mut idx = VectorIndex::new(8).unwrap();
        assert!(idx.upsert(1, &[0.0; 7]).is_err());
        assert!(idx.search(&[0.0; 9], 3, None).is_err());
        assert!(VectorIndex::new(0).is_err());
    }

    #[test]
    fn a_quarter_of_the_float_size() {
        let mut idx = VectorIndex::new(1024).unwrap();
        for (i, v) in vectors(100, 1024, 1).iter().enumerate() {
            idx.upsert(i as i64, v).unwrap();
        }
        assert_eq!(idx.bytes(), 100 * 1024 + 100 * 4);
    }
}
