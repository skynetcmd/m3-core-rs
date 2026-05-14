//! ONNX Runtime GLiNER NER backend (Phase 3c).
//!
//! Wraps the GLiNER model via the `ort` crate. Execution providers are
//! selected at init time in per-platform priority order. The model stays
//! GLiNER; only the runtime changes.
//!
//! The real `ort` dependency is gated behind the non-default `onnx` cargo
//! feature (it needs ONNX Runtime native libs). The default build still
//! provides the EP-priority table and span-decode logic — both pure Rust and
//! fully testable — plus an `OrtNer` whose `run` returns an error.

use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
use m3_error::{M3Error, Result};

/// Quantization variant for the loaded ONNX model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    Fp32,
    Fp16,
    Int8,
}

/// Target platform for execution-provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    LinuxNvidia,
    LinuxAmd,
    LinuxIntelGpu,
    Windows,
    MacOsAppleSilicon,
    CpuOnly,
}

/// Execution-provider priority list for a platform (§4b.2). The last entry is
/// always `"CPU"` as the universal fallback. `m3-core-py` may override this
/// from `M3_NER_EP_PRIORITY`.
pub fn ep_priority(platform: Platform) -> Vec<&'static str> {
    match platform {
        Platform::LinuxNvidia => vec!["CUDA", "TensorRT", "CPU"],
        Platform::LinuxAmd => vec!["ROCm", "CPU"],
        Platform::LinuxIntelGpu => vec!["OpenVINO", "CPU"],
        Platform::Windows => vec!["DirectML", "CPU"],
        Platform::MacOsAppleSilicon => vec!["CoreML", "CPU"],
        Platform::CpuOnly => vec!["CPU"],
    }
}

/// A decoded NER span: a labelled, scored `[start, end)` token range.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub label: usize,
    pub score: f32,
}

/// Decode a flattened GLiNER span-score tensor into a deduped, non-overlapping
/// span list.
///
/// `scores` is the row-major flattening of a `[max_spans, span_width, num_labels]`
/// tensor where `shape = (max_spans, span_width, num_labels)`. Index
/// `(s, w, l)` is the score that the span starting at token `s` with width
/// `w + 1` is an entity of label `l`.
///
/// Steps: threshold filter, then greedy overlap resolution (highest score
/// wins), then dedup of identical `[start, end, label)` spans.
pub fn decode_spans(
    scores: &[f32],
    shape: (usize, usize, usize),
    threshold: f32,
) -> Vec<Span> {
    let (max_spans, span_width, num_labels) = shape;
    let mut candidates: Vec<Span> = Vec::new();
    for s in 0..max_spans {
        for w in 0..span_width {
            for l in 0..num_labels {
                let idx = (s * span_width + w) * num_labels + l;
                let Some(&score) = scores.get(idx) else { continue };
                if score >= threshold {
                    candidates.push(Span { start: s, end: s + w + 1, label: l, score });
                }
            }
        }
    }

    // Highest score first so greedy resolution keeps the strongest spans.
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
    });

    let mut accepted: Vec<Span> = Vec::new();
    for cand in candidates {
        let overlaps = accepted
            .iter()
            .any(|a| cand.start < a.end && a.start < cand.end);
        if overlaps {
            continue;
        }
        // Dedup identical span+label (different widths could collide post-filter).
        let dup = accepted.iter().any(|a| {
            a.start == cand.start && a.end == cand.end && a.label == cand.label
        });
        if !dup {
            accepted.push(cand);
        }
    }

    accepted.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
    accepted
}

/// GLiNER NER backend over ONNX Runtime. The `ort::Session` + tokenizer live
/// here once the `onnx` feature is enabled; until then this is a config holder
/// whose `run` returns an error.
pub struct OrtNer {
    #[allow(dead_code)]
    model_path: String,
    #[allow(dead_code)]
    labels: Vec<String>,
    #[allow(dead_code)]
    threshold: f32,
    #[allow(dead_code)]
    quant: Quant,
    #[allow(dead_code)]
    ep_order: Vec<&'static str>,
}

impl OrtNer {
    pub fn new(
        model_path: impl Into<String>,
        labels: Vec<String>,
        threshold: f32,
        quant: Quant,
        platform: Platform,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            labels,
            threshold,
            quant,
            ep_order: ep_priority(platform),
        }
    }
}

impl ModelBackend for OrtNer {
    #[cfg(not(feature = "onnx"))]
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        Err(M3Error::Backend(
            "onnx backend not compiled — build with --features onnx".into(),
        ))
    }

    #[cfg(feature = "onnx")]
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        // TODO(onnx): tokenize batch.texts, build the ONNX input tensors,
        // run self.session with self.ep_order, return flattened span scores
        // as BatchOutput rows. Requires ONNX Runtime native libs.
        Err(M3Error::Backend(
            "onnx backend feature enabled but ort integration is a stub".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_priority_tables() {
        assert_eq!(ep_priority(Platform::LinuxNvidia), ["CUDA", "TensorRT", "CPU"]);
        assert_eq!(ep_priority(Platform::LinuxAmd), ["ROCm", "CPU"]);
        assert_eq!(ep_priority(Platform::LinuxIntelGpu), ["OpenVINO", "CPU"]);
        assert_eq!(ep_priority(Platform::Windows), ["DirectML", "CPU"]);
        assert_eq!(ep_priority(Platform::MacOsAppleSilicon), ["CoreML", "CPU"]);
        assert_eq!(ep_priority(Platform::CpuOnly), ["CPU"]);
        // CPU is always the final fallback.
        for p in [
            Platform::LinuxNvidia,
            Platform::LinuxAmd,
            Platform::LinuxIntelGpu,
            Platform::Windows,
            Platform::MacOsAppleSilicon,
            Platform::CpuOnly,
        ] {
            assert_eq!(*ep_priority(p).last().unwrap(), "CPU");
        }
    }

    #[test]
    fn decode_spans_threshold_filter() {
        // shape (2 spans, 1 width, 2 labels)
        let scores = vec![0.9, 0.1, 0.2, 0.8];
        let spans = decode_spans(&scores, (2, 1, 2), 0.5);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], Span { start: 0, end: 1, label: 0, score: 0.9 });
        assert_eq!(spans[1], Span { start: 1, end: 2, label: 1, score: 0.8 });
    }

    #[test]
    fn decode_spans_resolves_overlap_by_score() {
        // shape (3 spans, 2 widths, 1 label).
        // span(start=0,width=2) -> [0,2) score 0.95
        // span(start=1,width=1) -> [1,2) score 0.6  (overlaps, lower score -> dropped)
        // span(start=2,width=1) -> [2,3) score 0.7  (no overlap -> kept)
        let mut scores = vec![0.0; 3 * 2 * 1];
        scores[(0 * 2 + 1) * 1] = 0.95; // start 0, width idx 1 -> [0,2)
        scores[(1 * 2 + 0) * 1] = 0.6; // start 1, width idx 0 -> [1,2)
        scores[(2 * 2 + 0) * 1] = 0.7; // start 2, width idx 0 -> [2,3)
        let spans = decode_spans(&scores, (3, 2, 1), 0.5);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], Span { start: 0, end: 2, label: 0, score: 0.95 });
        assert_eq!(spans[1], Span { start: 2, end: 3, label: 0, score: 0.7 });
    }

    #[test]
    fn decode_spans_empty_when_all_below_threshold() {
        let scores = vec![0.1, 0.2, 0.3, 0.4];
        assert!(decode_spans(&scores, (2, 1, 2), 0.9).is_empty());
    }

    #[tokio::test]
    async fn ort_ner_is_stub_by_default() {
        let ner = OrtNer::new(
            "models/gliner/v1.onnx",
            vec!["PERSON".into(), "ORG".into()],
            0.5,
            Quant::Int8,
            Platform::Windows,
        );
        let err = ner.run(Batch::new(vec!["x".into()], 1)).await.unwrap_err();
        assert!(format!("{err}").contains("onnx backend not compiled"));
    }
}
