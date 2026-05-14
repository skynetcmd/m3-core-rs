use m3_graph::GraphIndex;

fn sample() -> GraphIndex {
    // a -> b -> c -> d ;  a -> e ;  b -> e
    let mut g = GraphIndex::new();
    g.add_edge("a", "b", "rel", 1.0);
    g.add_edge("b", "c", "rel", 1.0);
    g.add_edge("c", "d", "rel", 1.0);
    g.add_edge("a", "e", "rel", 1.0);
    g.add_edge("b", "e", "rel", 1.0);
    g
}

#[test]
fn add_node_idempotent() {
    let mut g = GraphIndex::new();
    g.add_node("x");
    g.add_node("x");
    assert_eq!(g.node_count(), 1);
}

#[test]
fn counts() {
    let g = sample();
    assert_eq!(g.node_count(), 5);
    assert_eq!(g.edge_count(), 5);
}

#[test]
fn neighbors_within_hop_distances() {
    let g = sample();
    let mut got = g.neighbors_within("a", 2);
    got.sort();
    assert_eq!(
        got,
        vec![
            ("a".to_string(), 0),
            ("b".to_string(), 1),
            ("c".to_string(), 2),
            ("e".to_string(), 1),
        ]
    );
}

#[test]
fn neighbors_within_unknown_start() {
    let g = sample();
    assert!(g.neighbors_within("zzz", 3).is_empty());
}

#[test]
fn shortest_path_basic() {
    let g = sample();
    assert_eq!(
        g.shortest_path("a", "d"),
        Some(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string()
        ])
    );
}

#[test]
fn shortest_path_self() {
    let g = sample();
    assert_eq!(g.shortest_path("a", "a"), Some(vec!["a".to_string()]));
}

#[test]
fn shortest_path_no_path() {
    let g = sample();
    // d has no outgoing edges, cannot reach a
    assert_eq!(g.shortest_path("d", "a"), None);
    assert_eq!(g.shortest_path("a", "missing"), None);
}

#[test]
fn expand_dedup_and_excludes_seeds() {
    let g = sample();
    // seeds a and b both reach e; result must dedup and not contain seeds
    let mut got = g.expand(&["a", "b"], 1, 100);
    got.sort();
    assert_eq!(got, vec!["c".to_string(), "e".to_string()]);
}

#[test]
fn expand_respects_limit() {
    let g = sample();
    let got = g.expand(&["a"], 3, 2);
    assert_eq!(got.len(), 2);
}
