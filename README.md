# faster-whisper-openai

OpenAI-compatible speech-to-text service built on `faster-whisper`, packaged as a container-first HTTP and WebSocket API with persistent cache mounts and GPU-friendly defaults.

## Highlights

- OpenAI-style `POST /v1/audio/transcriptions` endpoint
- Accepts `whisper-1` as an alias for the configured backend model
- Realtime WebSocket transcription at `GET /v1/realtime/transcriptions`
- Published multi-arch image for `linux/amd64` and `linux/arm64`
- Bind-mounted cache and temp storage for predictable container restarts
- Tunable model, batching, concurrency, cleanup, and proxy settings

## Image

- Repository: `ghcr.io/slime-turing/faster-whisper-openai`
- Default tag in Compose: `latest`
- Published by GitHub Actions

## Quick Start

### Docker Compose

```bash
cp .env.example .env
mkdir -p cache tmp
docker compose pull
docker compose up -d
```

Set `STT_UID` and `STT_GID` in `.env` to match the host user that owns `cache/` and `tmp/`.

### Docker Run

```bash
docker run --rm -p 9000:9000 \
  -e STT_MODEL=large-v3-turbo \
  -e STT_DEVICE=cpu \
  -v "$PWD/cache:/var/cache/stt" \
  -v "$PWD/tmp:/var/tmp/stt" \
  ghcr.io/slime-turing/faster-whisper-openai:latest
```

### Local Build Override

```bash
docker build -t faster-whisper-openai:dev .
IMAGE_NAME=faster-whisper-openai IMAGE_TAG=dev docker compose up -d
```

## Runtime Layout

- Cache root: `/var/cache/stt`
- Temp root: `/var/tmp/stt`
- Model cache: `/var/cache/stt/models`
- App port: `9000`

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

Common settings:

- `STT_UID`, `STT_GID`: host UID/GID that owns the bind-mounted `cache/` and `tmp/` directories
- `STT_MODEL`: faster-whisper model name to load at startup
- `STT_DEVICE`: `cuda`, `cpu`, or another `ctranslate2`-supported device
- `STT_COMPUTE_TYPE`: compute precision such as `float16`, `int8_float16`, or `int8`
- `STT_SKIP_MODEL_LOAD`: test-only flag that skips model download and initialization
- `STT_MAX_CONCURRENT_JOBS`: concurrent transcription jobs allowed at once
- `STT_LONG_FORM_SECONDS`: threshold after which batched inference is used
- `STT_WEBSOCKET_MAX_SECONDS`: maximum lifetime for a realtime session
- `STT_WEBSOCKET_AUTO_COMMIT_BYTES`: automatic realtime chunk flush threshold in bytes, default `262144` (256 KiB). Set `0` to disable auto-commit and require explicit client `commit` messages.
- `STT_TMP_TTL_HOURS`: retention window for temp files
- `STT_CLEANUP_INTERVAL_SECONDS`: background cleanup interval
- `http_proxy`, `https_proxy`, `no_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`: optional build/runtime proxy settings

Example:

```env
STT_WEBSOCKET_AUTO_COMMIT_BYTES=524288
```

That example auto-commits each realtime chunk after it grows to 512 KiB, even if the client does not send an explicit `commit` message first.

## Endpoints

- Swagger UI: `http://localhost:9000/docs`
- OpenAPI schema: `http://localhost:9000/openapi.json`
- Health endpoint: `http://localhost:9000/healthz`
- API docs: [docs/API.md](docs/API.md)

## Storage And Operations

- Bind-mounted cache ownership matters. If `STT_UID` and `STT_GID` do not match the host directory owner, model downloads can fail with permission errors.
- Cold starts after cache removal can take several minutes because the model has to be downloaded again.
- Use the included sync helper instead of raw `rsync --delete` when updating a remote deployment:

```bash
./scripts/sync-remote.sh user@example-host
```

The sync helper excludes `.env`, `cache/`, and `tmp/` so runtime state is not deleted during updates.

## Concurrency And Performance

The service now includes explicit concurrency and performance guardrails:

- Transcription work is bounded by `STT_MAX_CONCURRENT_JOBS`, so inference cannot fan out without limit.
- Temporary upload and realtime chunk files are tracked as active while in use, so background cleanup does not delete them mid-transcription.
- `/healthz` reads cached storage metrics instead of walking the cache and temp trees on every request.
- HTTP upload persistence is moved off the event loop with `asyncio.to_thread(...)`.
- Realtime audio is written to rotating chunk files on disk and transcribed incrementally across `commit` and `finish` events, instead of growing one in-memory buffer or one ever-growing temp file.
- Realtime chunk files can auto-commit once they reach `STT_WEBSOCKET_AUTO_COMMIT_BYTES`, which bounds per-chunk latency and disk growth for clients that stream bytes continuously.
- FastAPI JSON responses follow current best practice: plain return values plus response models, not deprecated global `ORJSONResponse` configuration.

## Testing

The smoke suite covers:

- disappearing files during `healthz`
- active temp-file cleanup protection
- cached health metrics
- realtime temp-file cleanup
- incremental transcript aggregation across WebSocket chunk commits
- websocket auto-commit once a chunk reaches the configured byte threshold
- dated production validation notes for live deployments in `docs/`

You can run the smoke tests in a disposable container:

```bash
docker run --rm -v "$PWD":/work -w /work python:3.12-slim bash -lc '
set -e
python -m pip install --upgrade pip >/tmp/pip-upgrade.log 2>&1
python -m pip install -r requirements.txt >/tmp/pip-install.log 2>&1
STT_SKIP_MODEL_LOAD=1 STT_CACHE_DIR=/tmp/faster-whisper-smoke-cache STT_TMP_DIR=/tmp/faster-whisper-smoke-tmp python -m unittest -q tests.test_api_smoke
'
```

Latest production validation note:

- [docs/production-validation-2026-03-08.md](docs/production-validation-2026-03-08.md)

## Project Policies

- License: [MIT](LICENSE)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)