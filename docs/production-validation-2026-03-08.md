# Production Validation 2026-03-08

This note captures a live GPU validation run against a production deployment.

## Environment

- Host: production deployment
- Container: `faster-whisper`
- Image: `ghcr.io/slime-turing/faster-whisper-openai:latest`
- GPU inside container: `NVIDIA GeForce RTX 4090`, driver `595.71`, `23028 MiB`
- Health status before testing: `ok`
- Runtime settings observed from the live container:
  - `STT_MODEL=large-v3-turbo`
  - `STT_DEVICE=cuda`
  - `STT_COMPUTE_TYPE=float16`
  - `STT_MAX_CONCURRENT_JOBS=2`
  - `STT_LONG_FORM_SECONDS=240`
  - `STT_WEBSOCKET_MAX_SECONDS=900`
  - `STT_WEBSOCKET_AUTO_COMMIT_BYTES` was unset in the container environment

## Test Fixtures

- `speech-short-fixed.wav`: 7.552s spoken English sample
- `speech-long-245s.wav`: 249.217s spoken English sample built by repeating the corrected short PCM payload
- `tone-245s.wav`: 245s synthetic tone-only WAV

The colocated TTS service produced a short WAV with an invalid `nframes` header (`2147483647`) even though the payload itself was valid PCM. The production STT service transcribed that short file successfully, but for repeatable long-form testing the payload had to be rewritten with a correct WAV header first.

## HTTP Results

| Case | Result | Observed behavior |
| --- | --- | --- |
| Short speech, JSON response | `200` in `0.343s` | Transcript matched the sample text and reported `duration_seconds=7.5520625` |
| Short speech, text response | `200` in `0.327s` | Plain text response worked as expected |
| Unsupported model (`bogus-model`) | `400` in `0.005s` | Returned `Only models large-v3-turbo, whisper-1 are available` |
| Empty upload | `400` in `0.003s` | Returned `Uploaded audio file is empty` |
| Long speech, JSON response | `200` in `7.479s` | Returned a repeated transcript and `duration_seconds=249.217375` |
| Long tone-only WAV | `400` in `1.841s` | Returned `Transcription produced no text` |

## Concurrency Results

Three long-form requests were launched at the same time against a deployment configured with `STT_MAX_CONCURRENT_JOBS=2`.

| Request | Result |
| --- | --- |
| 1 | `200` in `41.248s` |
| 2 | `200` in `48.809s` |
| 3 | `200` in `41.286s` |

Single long-form latency was about `7.5s`, so the live service slowed down sharply under parallel long-form load but still completed all three requests successfully. The observed behavior did not present as a clean fail-fast rejection or an obvious strict two-running-one-queued pattern from the client side. In practice, the concurrency limit prevented crashes, but GPU saturation and batching overhead under parallel long-form work were still substantial.

## WebSocket Results

Realtime validation was executed from inside the running container with the Python `websockets` client.

### Manual commit

- `start` returned `{"type":"ready","model":"large-v3-turbo"}`
- Sending `65536` bytes produced no message before `commit`
- Explicit `commit` returned a partial transcript:
  - text: `This is a faster whisper.`
  - language: `en`
  - duration: `1.3644375`
- `finish` returned the full final transcript with `duration_seconds=7.5520625`

### Auto-commit threshold

- A fresh session sent `300000` bytes without any explicit `commit`
- No partial message arrived within `15s`
- `finish` still returned the correct final transcript

The current local source sets a default `STT_WEBSOCKET_AUTO_COMMIT_BYTES=262144`, so this live result strongly suggests the production container is not running the newest websocket auto-commit implementation yet, even though it is using the same image tag name.

## Edge Cases And Notes

- Short malformed-but-decodable WAV input was accepted, but building long-form fixtures from the broken header failed until the WAV header was corrected.
- Tone-only input was rejected cleanly instead of returning garbage text.
- Unsupported models and empty uploads failed quickly with useful error messages.
- Manual websocket `commit` works correctly on the live deployment.
- Auto-commit behavior documented in the current source tree was not observed on the running production container.

## Operational Takeaways

- The deployment is healthy for normal short-form traffic and handles a 249s spoken sample quickly on the RTX 4090.
- Long-form parallel load is much slower than single-request long-form latency, so production sizing should treat `STT_MAX_CONCURRENT_JOBS=2` as an upper safety bound, not as a guarantee of near-linear throughput.
- If websocket auto-commit is required in production, the running container should be rebuilt and redeployed from the current source rather than relying on the existing `latest` tag already on the host.