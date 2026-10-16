# m3-embed-server

OpenAI-compatible CPU embedding server. Wraps the `m3-embed-llamacpp` in-process
backend behind an HTTP API on port 8082, intended as a fallback for the
m3-memory in-process embedder.

## Build

```powershell
cargo build -p m3-embed-server --release --features embedded
```

The binary lands at `target\release\m3-embed-server.exe`.

## Configuration

The server reads m3's two config files, the same ones m3 reads:

* `m3_config.toml`, found as m3 finds it: `--config <file>` > `M3_CONFIG_ROOT`
  > `M3_MEMORY_ROOT/config` > `~/.m3/config`.
* the secrets file it names (`[secrets] file`, default
  `<config root>/m3_secrets.toml`, user-only), for the token digest only.

Per setting: `"--set={key=value, ...}"` > `m3_config.toml` > the legacy
environment name (deprecated, warned) > the default. The startup log names
where every non-default value came from.

| key (m3_config.toml) | legacy env name | default |
|---|---|---|
| `[embed] gguf` | `M3_EMBED_GGUF` | none (required) |
| `[embed] ctx` | `M3_EMBED_CTX` | 8192 |
| `[embed.server] host` | `M3_EMBED_SERVER_HOST` | 127.0.0.1 |
| `[embed.server] port` | `M3_EMBED_SERVER_PORT` | 8082 |
| `[embed.server] streams` | `M3_EMBED_STREAMS` | 1 (CPU) / 2 (GPU builds) |
| `[embed.server] seq_max` | `M3_EMBED_SEQ_MAX` | 32 |
| `[embed.server] n_batch` | `M3_EMBED_N_BATCH` | `embed.ctx` |
| `[embed.server] n_ubatch` | `M3_EMBED_N_UBATCH` | `embed.ctx` |
| `[embed.server] coalesce_ms` | `M3_EMBED_COALESCE_MS` | 3 |
| `[embed.server] max_batch_tokens` | `M3_EMBED_MAX_BATCH_TOKENS` | 2048 |
| `[embed.server] remote_listen`, `tls_cert`, `tls_key` | none | unset |
| `[crypto] fips_mode`, `fips_strict` | `M3_FIPS_MODE`, `M3_FIPS_STRICT` | false |
| `[secrets] file` | none | `<config root>/m3_secrets.toml` |
| `M3_EMBED_SERVER_TOKEN_SHA256` in the secrets file `[secrets]` | none | unset |

The token digest is read only when `remote_listen` is set. A secrets file that
others can read is refused (Unix mode/owner; Windows DACL), as m3 refuses it.
Earlier versions' own `config.toml` is no longer read; if it is still present
the server says so once at startup (m3 migrates and removes it).

## Foreground / dev mode

```powershell
m3 config set embed.gguf "C:\path\to\bge-m3.gguf"
.\target\release\m3-embed-server.exe
# or, for one run:
.\target\release\m3-embed-server.exe "--set={embed.gguf=^C:\path\to\bge-m3.gguf^}"
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
.\m3-embed-server.exe --config "$HOME\.m3\config\m3_config.toml" install
```

What `install` does:
1. Checks that the service can start from that file alone (it will not see
   this shell's environment or a `--set`); a legacy environment value that
   differs from the file is refused with the `m3 config set` that records it.
2. Creates a service named `m3-embed-server` (auto-start, LocalSystem account)
   whose `binPath` is `<self>.exe --config <m3_config.toml> run-as-service`
   (LocalSystem cannot find your config root on its own). An existing
   registration with another command line is updated.

On macOS and Linux `install` writes a launchd agent / `systemd --user` unit
that runs `m3-embed-server --config <m3_config.toml>`, after the same check.

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

Stops the service if running, then removes it. `m3_config.toml` is m3's and
is left in place.

### Logs

Service-mode logs land in:

```
%PROGRAMDATA%\m3-embed-server\service.log.YYYY-MM-DD
```

Logs roll daily (UTC) via `tracing-appender`; the active day's file is the
newest one. `install` does **not** create any files in this directory until
the service first runs.

Tail the current day's log:

```powershell
Get-ChildItem $env:PROGRAMDATA\m3-embed-server\service.log.* |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Get-Content -Wait
```

Old rolled files are pruned automatically on service startup: anything older
than 14 days is deleted. No external rotator needed.

Foreground / dev mode keeps the original `env_logger` stderr behaviour — only
service mode uses the rolling-file appender.

### Recovery actions (restart on crash)

Configured automatically by `install` via a `sc.exe failure` call:

```
reset window: 60 s, actions: restart/5000/restart/5000/restart/5000
```

i.e. restart the service up to 3 times with a 5 s delay, resetting the
counter after 60 s without a failure. If the `sc.exe` call fails (rare —
usually missing elevation), `install` prints a WARN with the exact command
to run by hand and continues; the service is already registered at that
point.

## Follow-ups

- Non-Windows targets only support foreground mode. Adding systemd unit
  generation would be a parallel "sovereign install" story for Linux.
