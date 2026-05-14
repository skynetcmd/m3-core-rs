//! In-memory graph index and multi-hop traversal (Phase 4) plus reusable
//! resilience primitives (Phase 5).
//!
//! Maintains a compressed graph of memory relationships for recursive context
//! assembly. BFS/DFS over thousands of neighbors in microseconds.
//!
//! Phase 5 note: thread-safe SQLite connection pooling (plan §6.1) is deferred
//! to whichever crate owns DB access — it cannot live in this generic crate.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

pub mod resilience;
pub use resilience::{BreakerState, CircuitBreaker, RetryPolicy};

/// An edge in the relationship graph: a relationship kind and optional weight.
#[derive(Debug, Clone)]
pub struct Relationship {
    pub kind: String,
    pub weight: f32,
}

/// Compressed in-memory index over `memory_relationships`.
pub struct GraphIndex {
    graph: DiGraph<String, Relationship>,
    ids: HashMap<String, NodeIndex>,
}

impl GraphIndex {
    pub fn new() -> Self {
        GraphIndex {
            graph: DiGraph::new(),
            ids: HashMap::new(),
        }
    }

    /// Insert a node; idempotent — returns the existing index if already present.
    pub fn add_node(&mut self, id: &str) -> NodeIndex {
        if let Some(&idx) = self.ids.get(id) {
            return idx;
        }
        let idx = self.graph.add_node(id.to_string());
        self.ids.insert(id.to_string(), idx);
        idx
    }

    /// Add a directed edge; endpoints are created on demand (idempotent on nodes).
    pub fn add_edge(&mut self, from: &str, to: &str, kind: &str, weight: f32) {
        let f = self.add_node(from);
        let t = self.add_node(to);
        self.graph.add_edge(
            f,
            t,
            Relationship {
                kind: kind.to_string(),
                weight,
            },
        );
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// BFS from `start`, returning reachable node IDs with their hop distance.
    /// The start node itself is included at distance 0.
    pub fn neighbors_within(&self, start: &str, max_hops: usize) -> Vec<(String, usize)> {
        let Some(&start_idx) = self.ids.get(start) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start_idx);
        queue.push_back((start_idx, 0usize));
        while let Some((idx, dist)) = queue.pop_front() {
            out.push((self.graph[idx].clone(), dist));
            if dist == max_hops {
                continue;
            }
            for edge in self.graph.edges(idx) {
                let nbr = edge.target();
                if seen.insert(nbr) {
                    queue.push_back((nbr, dist + 1));
                }
            }
        }
        out
    }

    /// Unweighted shortest path (fewest hops) from `from` to `to`, inclusive of
    /// both endpoints. Returns `None` when no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let &from_idx = self.ids.get(from)?;
        let &to_idx = self.ids.get(to)?;
        if from_idx == to_idx {
            return Some(vec![from.to_string()]);
        }
        let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(from_idx);
        queue.push_back(from_idx);
        while let Some(idx) = queue.pop_front() {
            for edge in self.graph.edges(idx) {
                let nbr = edge.target();
                if seen.insert(nbr) {
                    prev.insert(nbr, idx);
                    if nbr == to_idx {
                        let mut path = vec![to_idx];
                        let mut cur = to_idx;
                        while let Some(&p) = prev.get(&cur) {
                            path.push(p);
                            cur = p;
                        }
                        path.reverse();
                        return Some(path.into_iter().map(|i| self.graph[i].clone()).collect());
                    }
                    queue.push_back(nbr);
                }
            }
        }
        None
    }

    /// Batch BFS from multiple seed nodes: dedups across seeds and caps at
    /// `limit`. Seeds themselves are excluded from the result — callers already
    /// hold them.
    ///
    /// In the real system this pairs with a single batched SQL fetch of the
    /// surrounding turns; this crate is only the in-memory index half.
    pub fn expand(&self, seeds: &[&str], max_hops: usize, limit: usize) -> Vec<String> {
        let seed_set: HashSet<&str> = seeds.iter().copied().collect();
        let mut out = Vec::new();
        let mut emitted = HashSet::new();
        for seed in seeds {
            for (id, _dist) in self.neighbors_within(seed, max_hops) {
                if seed_set.contains(id.as_str()) {
                    continue;
                }
                if emitted.insert(id.clone()) {
                    out.push(id);
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    }
}

impl Default for GraphIndex {
    fn default() -> Self {
        Self::new()
    }
}
