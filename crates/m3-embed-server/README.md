# m3-embed-server

OpenAI-compatible CPU embedding server. Wraps the `m3-embed-llamacpp` in-process
backend behind an HTTP API on port 8082, intended as a fallback for the
m3-memory in-process embedder.

## Build

```powershell
cargo build -p m3-embed-server --release --features embedded
```

The binary lands at `target\release\m3-embed-server.exe`.

## Foreground / dev mode

```powershell
$env:M3_EMBED_GGUF = "C:\Users\<USER>\.lmstudio\models\deepsweet\bge-m3-GGUF-Q4_K_M\bge-m3-GGUF-Q4_K_M.gguf"
.\target\release\m3-embed-server.exe
```

Ctrl-C to stop. Logs go to stderr.

Endpoints:
- `POST /embedding`         — llama-server shape
- `POST /v1/embeddings`     — OpenAI shape
- `GET  /health`            — `"OK\n"` once the dispatcher is up
- `GET  /metrics`           — JSON dispatcher stats

## Windows Service mode

The binary is its own service installer — no nssm, no external tooling.

### Install

```powershell
# In an *elevated* PowerShell:
$env:M3_EMBED_GGUF = "C:\path\to\model.gguf"
$env:M3_EMBED_SERVER_PORT = "8082"   # optional, defaults to 8082
.\m3-embed-server.exe install
```

What `install` does:
1. Connects to the Service Control Manager.
2. Creates a service named `m3-embed-server` (auto-start, LocalSystem account).
3. Points `binPath` at `<self>.exe run-as-service`.
4. Snapshots the current shell's `M3_EMBED_*` env vars into
   `%PROGRAMDATA%\m3-embed-server\config.toml` (LocalSystem cannot see your
   user env, so the service reads this file at start).

### Start / stop / status

```powershell
.\m3-embed-server.exe start     # or: sc start m3-embed-server
.\m3-embed-server.exe stop      # or: sc stop  m3-embed-server
.\m3-embed-server.exe status    # "running" / "stopped" / "not installed"
Get-Service m3-embed-server     # native PowerShell view
sc query m3-embed-server        # detailed state
```

`status` works without admin rights; `start` / `stop` need elevation.

### Uninstall

```powershell
# Elevated:
.\m3-embed-server.exe uninstall
```

Stops the service if running, then removes it. The config file
(`%PROGRAMDATA%\m3-embed-server\config.toml`) is left in place — delete it
manually if you want a clean uninstall.

### Logs

Service-mode logs land at:

```
%PROGRAMDATA%\m3-embed-server\service.log
```

Tail with:

```powershell
Get-Content -Wait $env:PROGRAMDATA\m3-embed-server\service.log
```

Log rotation is **not** built in — for long-running deployments, plan on an
external rotator (Task Scheduler running PowerShell `Compress-Archive` weekly,
or similar). See follow-ups below.

### Recovery actions (restart on crash)

The `windows-service` 0.8 crate does not expose `ChangeServiceConfig2`-style
recovery actions. After `install`, run this **once** (elevated) to set restart
on failure:

```powershell
sc.exe failure m3-embed-server reset= 60 actions= restart/5000/restart/5000/restart/5000
```

This restarts the service up to 3 times with a 5 s delay, resetting the
counter after 60 s without a failure.

## Config file format

`%PROGRAMDATA%\m3-embed-server\config.toml`:

```toml
[embed]
gguf = "C:/Users/<USER>/.lmstudio/models/deepsweet/bge-m3-GGUF-Q4_K_M/bge-m3-GGUF-Q4_K_M.gguf"
port = 8082
host = "127.0.0.1"
streams = 2
ctx = 8192
seq_max = 32
n_batch = 2048
n_ubatch = 512
coalesce_ms = 3
max_batch_tokens = 2048
```

All keys are optional except `gguf`. Env vars take precedence over file values
when both are set (handy for ad-hoc overrides in foreground mode).

## Follow-ups

- Recovery actions need a one-line `sc.exe failure` shim or a manual step
  (see above). `windows-service` 0.8 doesn't expose the API.
- Log rotation is not built in; use an external rotator or add the
  `tracing-appender` rolling-file crate in a future wave.
- Non-Windows targets only support foreground mode. Adding systemd unit
  generation would be a parallel "sovereign install" story for Linux.
