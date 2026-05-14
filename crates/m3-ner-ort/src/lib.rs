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

#[cfg(not(feature = "onnx"))]
use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
#[cfg(not(feature = "onnx"))]
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

// ============================================================================
// Default build (no `onnx` feature): `OrtNer` is a pure config holder whose
// `run` returns an error. No ONNX Runtime dependency.
// ============================================================================

/// GLiNER NER backend over ONNX Runtime. The `ort::Session` + tokenizer live
/// here once the `onnx` feature is enabled; until then this is a config holder
/// whose `run` returns an error.
#[cfg(not(feature = "onnx"))]
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

#[cfg(not(feature = "onnx"))]
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

#[cfg(not(feature = "onnx"))]
impl ModelBackend for OrtNer {
    async fn run(&self, _batch: Batch) -> Result<BatchOutput> {
        Err(M3Error::Backend(
            "onnx backend not compiled — build with --features onnx".into(),
        ))
    }
}

// ============================================================================
// `onnx` feature: real `ort` (ONNX Runtime) backend via `load-dynamic`.
//
// `ort` is built with the `load-dynamic` feature, so the ONNX Runtime native
// library is `dlopen`ed at runtime rather than linked at build time. The
// caller must point `ort` at an installed `onnxruntime` shared library:
//   - pass `Some(path)` as `dylib_path` to `OrtNer::new`, OR
//   - leave it `None` and set the `ORT_DYLIB_PATH` env var (read by `ort`).
// No absolute path is baked into this crate — it stays generic.
// ============================================================================

#[cfg(feature = "onnx")]
mod onnx_backend {
    use super::{decode_spans, ep_priority, Platform, Quant};
    use m3_dispatcher::{Batch, BatchOutput, ModelBackend};
    use m3_error::{M3Error, Result};
    use std::sync::Mutex;
    use tokenizers::Tokenizer;

    /// Initialise the `ort` environment, loading the ONNX Runtime dylib.
    ///
    /// If `dylib_path` is `Some`, the dylib at that path is loaded explicitly.
    /// If `None`, `ort` falls back to its own resolution (the `ORT_DYLIB_PATH`
    /// env var, then platform defaults). Idempotent — safe to call repeatedly;
    /// only the first call wins.
    fn ensure_ort_env(dylib_path: Option<&str>) -> Result<()> {
        match dylib_path {
            Some(p) => {
                // `init_from` loads the dylib at `p`, returning a builder.
                ort::init_from(p)
                    .map_err(|e| M3Error::Backend(format!("load onnxruntime dylib: {e}")))?
                    .with_name("m3-ner-ort")
                    .commit();
            }
            None => {
                // `ort` resolves the dylib itself (ORT_DYLIB_PATH / defaults).
                ort::init().with_name("m3-ner-ort").commit();
            }
        }
        Ok(())
    }

    /// GLiNER NER backend over ONNX Runtime.
    ///
    /// Holds a committed `ort::Session` plus the HF tokenizer. `Session::run`
    /// takes `&mut self`, but `ModelBackend::run` is `&self`, so the session
    /// sits behind a `Mutex` — inference is serialised per `OrtNer` instance
    /// (the dispatcher already fans out across multiple backend instances /
    /// streams for concurrency).
    pub struct OrtNer {
        session: Mutex<ort::session::Session>,
        tokenizer: Tokenizer,
        labels: Vec<String>,
        threshold: f32,
        #[allow(dead_code)]
        quant: Quant,
        #[allow(dead_code)]
        ep_order: Vec<&'static str>,
    }

    impl OrtNer {
        /// Load a GLiNER ONNX model and its tokenizer.
        ///
        /// - `model_path`: path to the exported GLiNER `.onnx` file.
        /// - `tokenizer_path`: path to the model's `tokenizer.json` (HF format).
        /// - `dylib_path`: optional explicit path to the `onnxruntime` shared
        ///   library; `None` defers to `ORT_DYLIB_PATH` / `ort`'s defaults.
        /// - `labels`: entity label set, in the order the model emits them.
        /// - `platform`: drives execution-provider priority via `ep_priority`.
        ///
        /// Execution providers are registered best-effort: each EP in the
        /// priority list is appended, and `ort` silently skips any that are
        /// unavailable at runtime (default `fail_silently` behaviour), so a
        /// machine with no CUDA/DirectML still loads on the CPU EP.
        pub fn new(
            model_path: impl AsRef<str>,
            tokenizer_path: impl AsRef<str>,
            dylib_path: Option<&str>,
            labels: Vec<String>,
            threshold: f32,
            quant: Quant,
            platform: Platform,
        ) -> Result<Self> {
            ensure_ort_env(dylib_path)?;

            let ep_order = ep_priority(platform);
            let eps = build_execution_providers(&ep_order);

            let mut builder = ort::session::Session::builder()
                .map_err(|e| M3Error::Backend(format!("ort session builder: {e}")))?;
            builder = builder
                .with_execution_providers(eps)
                .map_err(|e| M3Error::Backend(format!("ort register EPs: {e}")))?;
            let session = builder
                .commit_from_file(model_path.as_ref())
                .map_err(|e| M3Error::Backend(format!("ort load model: {e}")))?;

            let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
                .map_err(|e| M3Error::Backend(format!("tokenizer load: {e}")))?;

            Ok(Self {
                session: Mutex::new(session),
                tokenizer,
                labels,
                threshold,
                quant,
                ep_order,
            })
        }

        pub fn labels(&self) -> &[String] {
            &self.labels
        }
    }

    /// Map the `ep_priority` string list onto concrete `ort` EP dispatches.
    /// Only the EPs relevant to the target platforms are wired; the CPU EP is
    /// always appended last as the universal fallback. EPs that are not
    /// available at runtime are skipped by `ort` (fail-silently).
    fn build_execution_providers(
        ep_order: &[&'static str],
    ) -> Vec<ort::ep::ExecutionProviderDispatch> {
        let mut eps = Vec::new();
        for name in ep_order {
            match *name {
                "CUDA" => eps.push(ort::ep::CUDA::default().build()),
                "DirectML" => eps.push(ort::ep::DirectML::default().build()),
                "CPU" => eps.push(ort::ep::CPU::default().build()),
                // TensorRT/ROCm/OpenVINO/CoreML: not wired here — the CPU
                // fallback below still applies. Add as needed.
                _ => {}
            }
        }
        if !ep_order.contains(&"CPU") {
            eps.push(ort::ep::CPU::default().build());
        }
        eps
    }

    impl ModelBackend for OrtNer {
        async fn run(&self, batch: Batch) -> Result<BatchOutput> {
            let mut rows = Vec::with_capacity(batch.texts.len());
            for text in &batch.texts {
                rows.push(self.run_one(text)?);
            }
            Ok(BatchOutput::new(rows))
        }
    }

    impl OrtNer {
        /// Full single-text inference path: tokenize -> build input tensors ->
        /// session run -> flatten the span-score output into one `Vec<f32>`.
        ///
        /// NOTE: this path is structurally complete but has NOT been run
        /// end-to-end — it needs a real exported GLiNER ONNX model plus its
        /// `tokenizer.json`, neither of which ships in this repo. The exact
        /// input tensor names (`input_ids`, `attention_mask`, ...) and the
        /// output rank/shape depend on the specific GLiNER export and may need
        /// adjusting once a real model is available.
        fn run_one(&self, text: &str) -> Result<Vec<f32>> {
            let encoding = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| M3Error::Backend(format!("tokenize: {e}")))?;

            let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
            let mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| m as i64)
                .collect();
            let seq_len = ids.len();

            // GLiNER ONNX exports typically expect `[batch, seq_len]` int64
            // tensors. Batch dim is 1 here; the dispatcher coalesces at a
            // higher level.
            let input_ids = ort::value::Tensor::from_array(([1usize, seq_len], ids))
                .map_err(|e| M3Error::Backend(format!("input_ids tensor: {e}")))?;
            let attention_mask = ort::value::Tensor::from_array(([1usize, seq_len], mask))
                .map_err(|e| M3Error::Backend(format!("attention_mask tensor: {e}")))?;

            let mut session = self
                .session
                .lock()
                .map_err(|_| M3Error::Backend("ort session mutex poisoned".into()))?;

            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention_mask,
                ])
                .map_err(|e| M3Error::Backend(format!("ort inference: {e}")))?;

            // GLiNER's span-score output is the first output tensor. Extract
            // it as a flat f32 slice; the consumer reshapes via `decode_spans`
            // using the known `(max_spans, span_width, num_labels)` geometry.
            let first_output = outputs
                .iter()
                .next()
                .ok_or_else(|| M3Error::Backend("ort produced no outputs".into()))?;
            let (_shape, scores) = first_output
                .1
                .try_extract_tensor::<f32>()
                .map_err(|e| M3Error::Backend(format!("extract span scores: {e}")))?;

            Ok(scores.to_vec())
        }
    }

    /// Re-export of `decode_spans` so callers of the onnx backend can turn the
    /// flat `BatchOutput` rows back into spans without reaching outside the
    /// module. Geometry is model-specific and supplied by the caller.
    pub fn decode_row(
        scores: &[f32],
        shape: (usize, usize, usize),
        threshold: f32,
    ) -> Vec<super::Span> {
        decode_spans(scores, shape, threshold)
    }

    /// Expose the configured score threshold (used by `decode_row` callers).
    impl OrtNer {
        pub fn threshold(&self) -> f32 {
            self.threshold
        }
    }
}

#[cfg(feature = "onnx")]
pub use onnx_backend::{decode_row, OrtNer};

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

    #[cfg(not(feature = "onnx"))]
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

    // Proves the `load-dynamic` linkage actually reaches a real ONNX Runtime
    // DLL at runtime. Set `ORT_DYLIB_PATH` to the installed `onnxruntime`
    // shared library before running, e.g. on this machine:
    //   $env:ORT_DYLIB_PATH = "C:\Users\...\onnxruntime\capi\onnxruntime.dll"
    //   cargo test -p m3-ner-ort --features onnx -- --ignored ort_dylib_loads
    // Ignored by default because it depends on an env var being set.
    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "requires ORT_DYLIB_PATH pointing at a real onnxruntime DLL"]
    fn ort_dylib_loads() {
        // `init_from` loads the dylib explicitly; `init` would read
        // ORT_DYLIB_PATH. Either way, building a SessionBuilder forces the
        // dylib to be resolved and its API table bound.
        let path = std::env::var("ORT_DYLIB_PATH")
            .expect("set ORT_DYLIB_PATH to the onnxruntime DLL");
        ort::init_from(&path)
            .expect("ort::init_from failed to load the onnxruntime DLL")
            .with_name("m3-ner-ort-test")
            .commit();
        // Creating a SessionBuilder needs the runtime's C API to be live.
        ort::session::Session::builder()
            .expect("ort::Session::builder() failed — DLL not loaded / API mismatch");
    }
}
