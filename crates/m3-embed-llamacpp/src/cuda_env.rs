//! Process environment that llama.cpp's CUDA backend reads at init.
//!
//! Today that is one variable: `GGML_CUDA_DISABLE_GRAPHS`. On Blackwell
//! (compute capability 12.0+) with the 595.xx drivers, CUDA Graphs raise embed
//! latency from ~8 ms to ~2100 ms (~260x); disabling them costs ~5-10%
//! throughput on older GPUs. ggml-cuda reads the variable once, at its first
//! graph evaluation, so it must be set before the first decode.
//!
//! The parsing half is compiled on every build so it is unit-tested without a
//! GPU; only the probe and the `set_var` are CUDA-only.

#![cfg_attr(not(feature = "embedded-cuda"), allow(dead_code))]

#[cfg(feature = "embedded-cuda")]
const VAR: &str = "GGML_CUDA_DISABLE_GRAPHS";

/// Set the process environment llama.cpp reads at init (CUDA builds only; a
/// no-op elsewhere). Idempotent: the probe runs once per process.
///
/// ⚠ It may call `std::env::set_var`. Call it at the earliest point the host
/// has — `main` before any thread starts (the embed server), or module import
/// (m3-core-py). In a process that already has threads (any Python process)
/// a concurrent C `getenv` on another thread races the write; glibc does not
/// lock the environment. The window is one call at import time, before any
/// llama.cpp thread exists, and there is no API alternative: ggml-cuda only
/// reads the variable.
#[cfg(feature = "embedded")]
pub fn prepare_process_env() {
    static PREPARED: std::sync::Once = std::sync::Once::new();
    PREPARED.call_once(|| {
        #[cfg(feature = "embedded-cuda")]
        disable_cuda_graphs_on_blackwell();
    });
}

#[cfg(feature = "embedded-cuda")]
fn disable_cuda_graphs_on_blackwell() {
    // An operator's explicit value (any value) wins.
    if std::env::var_os(VAR).is_some() {
        log::info!("{VAR} is already set; leaving it as the operator chose");
        return;
    }
    match detect_cuda_compute_caps() {
        Ok(gpus) => {
            for (i, cap) in &gpus {
                log::info!("CUDA device {i}: compute capability {cap:.1}");
            }
            match gpus.iter().find(|(_, cap)| *cap >= 12.0) {
                Some((i, cap)) => {
                    log::info!(
                        "CUDA device {i} is compute {cap:.1} (Blackwell or newer): setting {VAR}=1 \
                         to avoid the CUDA Graphs latency pathology"
                    );
                    std::env::set_var(VAR, "1");
                }
                None => log::info!("no CUDA device at compute 12.0+; CUDA Graphs stay enabled"),
            }
        }
        Err(cause) => {
            log::warn!(
                "could not detect the CUDA compute capability ({cause}); setting {VAR}=1 \
                 defensively — costs ~5-10% throughput on pre-Blackwell GPUs. Set {VAR} yourself \
                 to choose"
            );
            std::env::set_var(VAR, "1");
        }
    }
}

/// `(device index, compute capability)` for every GPU `nvidia-smi` lists, or
/// why it could not be read (spawn failure, exit status + stderr, encoding,
/// or the unparseable line).
#[cfg(feature = "embedded-cuda")]
pub(crate) fn detect_cuda_compute_caps() -> Result<Vec<(usize, f32)>, String> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,compute_cap", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("could not run nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "nvidia-smi exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(out.stdout).map_err(|e| format!("nvidia-smi output is not UTF-8: {e}"))?;
    parse_compute_caps(&stdout)
}

/// Parse `nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader`.
pub(crate) fn parse_compute_caps(stdout: &str) -> Result<Vec<(usize, f32)>, String> {
    let mut gpus = Vec::new();
    for (n, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (idx, cap) = line
            .split_once(',')
            .ok_or_else(|| format!("nvidia-smi line {}: expected `index, compute_cap`, got {line:?}", n + 1))?;
        let idx = idx
            .trim()
            .parse::<usize>()
            .map_err(|e| format!("nvidia-smi line {}: GPU index {:?} is not a number: {e}", n + 1, idx.trim()))?;
        let cap = cap
            .trim()
            .parse::<f32>()
            .map_err(|e| format!("nvidia-smi line {}: compute capability {:?} is not a number: {e}", n + 1, cap.trim()))?;
        gpus.push((idx, cap));
    }
    if gpus.is_empty() {
        return Err("nvidia-smi listed no GPUs".into());
    }
    Ok(gpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_gpu_with_its_index() {
        let got = parse_compute_caps("0, 8.9\n1, 12.0\n").unwrap();
        assert_eq!(got, vec![(0, 8.9), (1, 12.0)]);
    }

    #[test]
    fn each_failure_names_its_cause() {
        assert!(parse_compute_caps("").unwrap_err().contains("no GPUs"));
        assert!(parse_compute_caps("8.9").unwrap_err().contains("line 1"));
        assert!(parse_compute_caps("0, [N/A]").unwrap_err().contains("compute capability"));
        assert!(parse_compute_caps("x, 8.9").unwrap_err().contains("GPU index"));
    }
}
