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
- `model`: optional model name, defaults to the configured server model
- `language`: optional language hint
- `prompt`: optional initial prompt
- `response_format`: optional, `json` or `text`; defaults to `json`

Example request:

```bash
curl -X POST http://localhost:9001/v1/audio/transcriptions \
  -F model=large-v3-turbo \
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

### Error Response

```json
{
  "detail": "Only model large-v3-turbo is available"
}
```

Status codes:

- `200`: transcription succeeded
- `400`: invalid upload, unsupported model, or no transcription text produced

## WebSocket Transcription

### `GET /v1/realtime/transcriptions`

Client sends JSON control messages and binary audio chunks.

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

Binary frames are treated as raw audio payload for the current buffer.