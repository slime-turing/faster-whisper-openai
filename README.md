# faster-whisper-openai

OpenAI-compatible speech-to-text service built on `faster-whisper`, with HTTP and WebSocket transcription endpoints, bind-mounted cache directories, and container-first deployment.

## Features

- OpenAI-style `POST /v1/audio/transcriptions` endpoint
- WebSocket transcription stream at `GET /v1/realtime/transcriptions`
- Configurable model, device, batching, and cleanup behavior through environment variables
- Persistent model cache and temp storage through bind mounts
- Multi-arch container publishing to GHCR for `linux/amd64` and `linux/arm64`

## Quick Start

### Docker Compose

```bash
cp .env.example .env
mkdir -p cache tmp
docker compose up -d
```

Set `STT_UID` and `STT_GID` in `.env` to match the host user that owns `cache/` and `tmp/`.

### Docker Run

```bash
docker run --rm -p 9001:9000 \
	-e STT_MODEL=large-v3-turbo \
	-e STT_DEVICE=cpu \
	-v "$PWD/cache:/var/cache/stt" \
	-v "$PWD/tmp:/var/tmp/stt" \
	ghcr.io/slime-turing/faster-whisper-openai:latest
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

Important settings:

- `STT_UID`, `STT_GID`: host UID/GID that owns the bind-mounted `cache/` and `tmp/` directories
- `STT_MODEL`: faster-whisper model name to load at startup
- `STT_DEVICE`: `cuda`, `cpu`, or another `ctranslate2`-supported device
- `STT_COMPUTE_TYPE`: compute precision such as `float16`, `int8_float16`, or `int8`
- `STT_SKIP_MODEL_LOAD`: test-only flag that skips model download and initialization at startup
- `STT_MAX_CONCURRENT_JOBS`: concurrent transcription jobs allowed at once
- `STT_LONG_FORM_SECONDS`: threshold after which batched inference is used
- `http_proxy`, `https_proxy`, `no_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`: optional build/runtime proxy settings

## Safe Remote Sync

Use the included sync helper instead of a raw `rsync --delete` from another checkout:

```bash
./scripts/sync-remote.sh user@example-host
```

The script excludes `.env`, `cache/`, and `tmp/` so runtime state is not deleted during updates.

## API

Human-readable endpoint and message documentation lives in [docs/API.md](docs/API.md).

The service also exposes:

- Swagger UI: `http://localhost:9001/docs`
- OpenAPI schema: `http://localhost:9001/openapi.json`
- Health endpoint: `http://localhost:9001/healthz`

## Image Publishing

GitHub Actions builds and publishes multi-arch images to GHCR.

- Repository image: `ghcr.io/slime-turing/faster-whisper-openai`
- Published platforms: `linux/amd64`, `linux/arm64`

## Notes

- Bind-mounted cache ownership matters. If `STT_UID` and `STT_GID` do not match the host directory owner, model downloads can fail with permission errors.
- Cold starts after cache removal can take several minutes because the model has to be downloaded again.

## Project Policies

- License: [MIT](LICENSE)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)