# API Reference

## Health

### `GET /healthz`

Returns process and cache status.

Example response:

```json
{
  "status": "ok",
  "model": "large-v3-turbo",
  "device": "cuda",
  "compute_type": "float16",
  "active_jobs": 0,
  "max_concurrent_jobs": 2,
  "uptime_seconds": 123,
  "cache_size_bytes": 987654321
}
```

## HTTP Transcription

### `POST /v1/audio/transcriptions`

Multipart form fields:

- `file`: required binary audio upload
- `model`: optional model name, defaults to the configured server model; `whisper-1` is accepted as an OpenAI-compatible alias
- `language`: optional language hint
- `prompt`: optional initial prompt
- `response_format`: optional, `json` or `text`; defaults to `json`

Example request:

```bash
curl -X POST http://localhost:9000/v1/audio/transcriptions \
  -F model=whisper-1 \
  -F file=@sample.ogg
```

### JSON Response

```json
{
  "text": "transcribed text",
  "model": "large-v3-turbo",
  "language": "en",
  "duration_seconds": 3.42
}
```

### Plain Text Response

When `response_format=text`, the response body is plain text instead of JSON.

Example:

```text
transcribed text
```

### Error Response

```json
{
  "detail": "Only models large-v3-turbo, whisper-1 are available"
}
```

Status codes:

- `200`: transcription succeeded
- `400`: invalid upload, unsupported model, or no transcription text produced

## WebSocket Transcription

### `GET /v1/realtime/transcriptions`

Client sends JSON control messages and binary audio chunks.

Server-side realtime chunk behavior:

- Binary frames are appended to the current chunk file on disk.
- The server auto-commits a chunk once it reaches `STT_WEBSOCKET_AUTO_COMMIT_BYTES`.
- Default auto-commit threshold: `262144` bytes (`256 KiB`).
- Set `STT_WEBSOCKET_AUTO_COMMIT_BYTES=0` to disable auto-commit and require explicit client `commit` messages.

Example configuration:

```env
STT_WEBSOCKET_AUTO_COMMIT_BYTES=524288
```

That example auto-commits each chunk after it reaches `512 KiB`.

#### Client control messages

Start message:

```json
{
  "type": "start",
  "language": "en",
  "prompt": "domain-specific hint",
  "suffix": ".wav"
}
```

Commit current buffer:

```json
{
  "type": "commit"
}
```

Finish stream and close connection:

```json
{
  "type": "finish"
}
```

Reset buffered audio:

```json
{
  "type": "reset"
}
```

#### Server messages

Ready:

```json
{
  "type": "ready",
  "model": "large-v3-turbo"
}
```

Partial transcript:

```json
{
  "type": "partial",
  "text": "hello world",
  "language": "en",
  "duration_seconds": 1.25,
  "model": "large-v3-turbo",
  "final": false
}
```

Final transcript:

```json
{
  "type": "final",
  "text": "hello world",
  "language": "en",
  "duration_seconds": 2.7,
  "model": "large-v3-turbo",
  "final": true
}
```

Reset acknowledgement:

```json
{
  "type": "reset"
}
```

Error:

```json
{
  "type": "error",
  "detail": "unsupported action: pause"
}
```

Binary frames are treated as raw audio payload for the current chunk.