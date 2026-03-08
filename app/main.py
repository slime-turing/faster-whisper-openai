from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("STT_MODEL", "large-v3-turbo")
    device: str = os.getenv("STT_DEVICE", "cuda")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "float16")
    skip_model_load: bool = getenv_bool("STT_SKIP_MODEL_LOAD", False)
    cpu_threads: int = getenv_int("STT_CPU_THREADS", 4)
    num_workers: int = getenv_int("STT_NUM_WORKERS", 2)
    beam_size: int = getenv_int("STT_BEAM_SIZE", 5)
    best_of: int = getenv_int("STT_BEST_OF", 5)
    batch_size: int = getenv_int("STT_BATCH_SIZE", 8)
    max_concurrent_jobs: int = getenv_int("STT_MAX_CONCURRENT_JOBS", 2)
    long_form_seconds: int = getenv_int("STT_LONG_FORM_SECONDS", 240)
    websocket_max_seconds: int = getenv_int("STT_WEBSOCKET_MAX_SECONDS", 900)
    websocket_auto_commit_bytes: int = getenv_int("STT_WEBSOCKET_AUTO_COMMIT_BYTES", 262144)
    tmp_ttl_hours: int = getenv_int("STT_TMP_TTL_HOURS", 6)
    cache_max_gb: float = getenv_float("STT_CACHE_MAX_GB", 30.0)
    cleanup_interval_seconds: int = getenv_int("STT_CLEANUP_INTERVAL_SECONDS", 3600)
    vad_filter: bool = getenv_bool("STT_VAD_FILTER", True)
    allow_origins: list[str] = field(
        default_factory=lambda: [
            value.strip()
            for value in os.getenv("STT_ALLOW_ORIGINS", "*").split(",")
            if value.strip()
        ]
    )
    cache_dir: Path = Path(os.getenv("STT_CACHE_DIR", "/var/cache/stt"))
    tmp_dir: Path = Path(os.getenv("STT_TMP_DIR", "/var/tmp/stt"))


@dataclass
class StorageMetrics:
    cache_size_bytes: int = 0
    tmp_size_bytes: int = 0

    @property
    def total_size_bytes(self) -> int:
        return self.cache_size_bytes + self.tmp_size_bytes


settings = Settings()
CONFIGURED_MODEL = settings.model.split("/")[-1].strip() or settings.model
MODEL_ALIASES = {CONFIGURED_MODEL, "whisper-1"}


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    compute_type: str
    active_jobs: int
    max_concurrent_jobs: int
    uptime_seconds: int
    cache_size_bytes: int


class TranscriptionResponse(BaseModel):
    text: str
    model: str
    language: str | None
    duration_seconds: float


def _build_incremental_prompt(prompt: str | None, transcript_parts: list[str]) -> str | None:
    parts = [part for part in transcript_parts if part]
    if prompt:
        parts.insert(0, prompt)
    if not parts:
        return None
    return "\n".join(parts)[-2048:]


app = FastAPI(title="Faster Whisper OpenAI Wrapper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_queue_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
_loaded_at = time.time()
_active_jobs = 0
_cleanup_task: asyncio.Task | None = None
_active_tmp_files: set[Path] = set()
_active_tmp_files_lock = threading.Lock()
_storage_metrics = StorageMetrics()
_storage_metrics_lock = threading.Lock()


def _ensure_dirs() -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)


def _mark_tmp_file_active(path: Path) -> None:
    with _active_tmp_files_lock:
        _active_tmp_files.add(path)


def _unmark_tmp_file_active(path: Path) -> None:
    with _active_tmp_files_lock:
        _active_tmp_files.discard(path)


def _is_tmp_file_active(path: Path) -> bool:
    with _active_tmp_files_lock:
        return path in _active_tmp_files


def _set_storage_metrics(*, cache_size_bytes: int, tmp_size_bytes: int) -> None:
    with _storage_metrics_lock:
        _storage_metrics.cache_size_bytes = cache_size_bytes
        _storage_metrics.tmp_size_bytes = tmp_size_bytes


def _bump_tmp_size_bytes(delta: int) -> None:
    with _storage_metrics_lock:
        _storage_metrics.tmp_size_bytes = max(0, _storage_metrics.tmp_size_bytes + delta)


def _snapshot_storage_metrics() -> StorageMetrics:
    with _storage_metrics_lock:
        return StorageMetrics(
            cache_size_bytes=_storage_metrics.cache_size_bytes,
            tmp_size_bytes=_storage_metrics.tmp_size_bytes,
        )


def _load_model() -> tuple[WhisperModel, BatchedInferencePipeline]:
    model = WhisperModel(
        settings.model,
        device=settings.device,
        compute_type=settings.compute_type,
        cpu_threads=settings.cpu_threads,
        num_workers=settings.num_workers,
        download_root=str(settings.cache_dir / "models"),
    )
    pipeline = BatchedInferencePipeline(model=model)
    return model, pipeline


_ensure_dirs()
if settings.skip_model_load:
    _model, _pipeline = None, None
else:
    _model, _pipeline = _load_model()


@contextlib.asynccontextmanager
async def limited_job() -> AsyncIterator[None]:
    global _active_jobs
    await _queue_semaphore.acquire()
    _active_jobs += 1
    try:
        yield
    finally:
        _active_jobs -= 1
        _queue_semaphore.release()


def _cleanup_tmp_files(now: float) -> None:
    max_age_seconds = settings.tmp_ttl_hours * 3600
    cutoff = now - max_age_seconds
    for path in settings.tmp_dir.rglob("*"):
        if not path.exists() or not path.is_file():
            continue
        try:
            if _is_tmp_file_active(path):
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


def _cleanup_model_cache() -> dict[str, Any]:
    cache_root = settings.cache_dir / "models"
    if not cache_root.exists():
        return {"deleted_revisions": 0, "freed_bytes": 0}

    try:
        cache_info = scan_cache_dir(cache_dir=cache_root)
    except Exception:
        return {"deleted_revisions": 0, "freed_bytes": 0}

    total_bytes = cache_info.size_on_disk
    limit_bytes = int(settings.cache_max_gb * 1024 * 1024 * 1024)
    if total_bytes <= limit_bytes:
        return {"deleted_revisions": 0, "freed_bytes": 0}

    stale_revisions: list[str] = []
    repos = sorted(cache_info.repos, key=lambda repo: repo.last_accessed or repo.last_modified or 0)
    for repo in repos:
        repo_id = getattr(repo, "repo_id", "") or ""
        keep_repo = settings.model in repo_id or repo_id.endswith(settings.model)
        if keep_repo:
                        continue
        for revision in repo.revisions:
            stale_revisions.append(revision.commit_hash)
        if stale_revisions:
            break

    if not stale_revisions:
        return {"deleted_revisions": 0, "freed_bytes": 0}

    strategy = cache_info.delete_revisions(*stale_revisions)
    strategy.execute()
    return {
        "deleted_revisions": len(stale_revisions),
        "freed_bytes": strategy.expected_freed_size,
    }


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        now = time.time()
        await asyncio.to_thread(_cleanup_tmp_files, now)
        await asyncio.to_thread(_cleanup_model_cache)
        await asyncio.to_thread(_refresh_storage_metrics)


def _create_tmp_path(prefix: str, suffix: str) -> Path:
    temp_path = settings.tmp_dir / f"{prefix}-{uuid.uuid4().hex}{suffix}"
    temp_path.touch()
    _mark_tmp_file_active(temp_path)
    return temp_path


def _delete_tmp_path(path: Path) -> None:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    path.unlink(missing_ok=True)
    _unmark_tmp_file_active(path)
    if size:
        _bump_tmp_size_bytes(-size)


def _append_bytes_to_path(path: Path, data: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(data)
    _bump_tmp_size_bytes(len(data))


def _path_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _save_upload_to_path(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "audio").suffix or ".bin"
    temp_path = _create_tmp_path("upload", suffix)
    with temp_path.open("wb") as handle:
        shutil.copyfileobj(upload_file.file, handle)
    try:
        size = temp_path.stat().st_size
    except FileNotFoundError:
        size = 0
    if size:
        _bump_tmp_size_bytes(size)
    return temp_path


def _normalize_requested_model(model_name: str | None) -> str:
    requested = (model_name or settings.model).strip()
    if not requested:
        return CONFIGURED_MODEL
    if "/" in requested:
        requested = requested.split("/")[-1]
    if requested in MODEL_ALIASES:
        return CONFIGURED_MODEL
    return requested


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _refresh_storage_metrics() -> StorageMetrics:
    metrics = StorageMetrics(
        cache_size_bytes=_dir_size_bytes(settings.cache_dir) if settings.cache_dir.exists() else 0,
        tmp_size_bytes=_dir_size_bytes(settings.tmp_dir) if settings.tmp_dir.exists() else 0,
    )
    _set_storage_metrics(
        cache_size_bytes=metrics.cache_size_bytes,
        tmp_size_bytes=metrics.tmp_size_bytes,
    )
    return metrics


def _transcribe_file(path: Path, language: str | None, prompt: str | None) -> dict[str, Any]:
    if _model is None or _pipeline is None:
        raise RuntimeError("Model is not loaded")

    transcribe_kwargs = {
        "beam_size": settings.beam_size,
        "best_of": settings.best_of,
        "vad_filter": settings.vad_filter,
        "word_timestamps": False,
        "condition_on_previous_text": True,
        "initial_prompt": prompt or None,
        "language": language or None,
    }
    if path.stat().st_size <= 0:
        raise ValueError("Uploaded audio file is empty")

    audio_seconds = 0.0
    info = None
    text = ""
    if path.stat().st_size > 0:
        try:
            segments, info = _model.transcribe(str(path), without_timestamps=True, **transcribe_kwargs)
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            if not text and info is None:
                raise ValueError("Transcription produced no text")
            audio_seconds = float(getattr(info, "duration", 0.0) or 0.0)
            if audio_seconds >= settings.long_form_seconds:
                batched_segments, info = _pipeline.transcribe(str(path), batch_size=settings.batch_size, **transcribe_kwargs)
                text = " ".join(segment.text.strip() for segment in batched_segments if segment.text.strip()).strip()
                audio_seconds = float(getattr(info, "duration", audio_seconds) or audio_seconds)
        except TypeError:
            batched_segments, info = _pipeline.transcribe(str(path), batch_size=settings.batch_size, **transcribe_kwargs)
            text = " ".join(segment.text.strip() for segment in batched_segments if segment.text.strip()).strip()
            audio_seconds = float(getattr(info, "duration", 0.0) or 0.0)

    if not text:
        raise ValueError("Transcription produced no text")

    return {
        "text": text,
        "language": getattr(info, "language", language or None),
        "duration_seconds": audio_seconds,
    }


async def transcribe_path(path: Path, language: str | None, prompt: str | None) -> dict[str, Any]:
    async with limited_job():
        return await asyncio.to_thread(_transcribe_file, path, language, prompt)


@app.on_event("startup")
async def on_startup() -> None:
    global _cleanup_task
    await asyncio.to_thread(_cleanup_tmp_files, time.time())
    await asyncio.to_thread(_cleanup_model_cache)
    await asyncio.to_thread(_refresh_storage_metrics)
    _cleanup_task = asyncio.create_task(cleanup_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        _cleanup_task = None


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    metrics = _snapshot_storage_metrics()
    return HealthResponse(
        status="ok",
        model=settings.model,
        device=settings.device,
        compute_type=settings.compute_type,
        active_jobs=_active_jobs,
        max_concurrent_jobs=settings.max_concurrent_jobs,
        uptime_seconds=int(time.time() - _loaded_at),
        cache_size_bytes=metrics.total_size_bytes,
    )


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default=settings.model),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
) -> TranscriptionResponse | PlainTextResponse:
    requested_model = _normalize_requested_model(model)
    if requested_model and requested_model != CONFIGURED_MODEL:
        supported_models = ", ".join(sorted(MODEL_ALIASES))
        raise HTTPException(status_code=400, detail=f"Only models {supported_models} are available")

    temp_path = await asyncio.to_thread(_save_upload_to_path, file)
    try:
        result = await transcribe_path(temp_path, language, prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await asyncio.to_thread(_delete_tmp_path, temp_path)

    payload = TranscriptionResponse(
        text=result["text"],
        model=settings.model,
        language=result["language"],
        duration_seconds=result["duration_seconds"],
    )
    if response_format == "text":
        return PlainTextResponse(content=payload.text)
    return payload


@app.websocket("/v1/realtime/transcriptions")
async def websocket_transcriptions(websocket: WebSocket) -> None:
    await websocket.accept()
    suffix = ".wav"
    language: str | None = None
    prompt: str | None = None
    started_at = time.time()
    chunk_path: Path | None = None
    transcript_parts: list[str] = []
    total_duration_seconds = 0.0

    async def ensure_chunk_path() -> Path:
        nonlocal chunk_path
        if chunk_path is None:
            chunk_path = await asyncio.to_thread(_create_tmp_path, "stream", suffix or ".wav")
        return chunk_path

    async def flush_buffer(final: bool = False) -> None:
        nonlocal chunk_path, total_duration_seconds
        message_type = "final" if final else "partial"
        if chunk_path is None:
            combined_text = " ".join(part for part in transcript_parts if part).strip()
            await websocket.send_bytes(
                orjson.dumps(
                    {
                        "type": message_type,
                        "text": combined_text if final else "",
                        "language": language,
                        "duration_seconds": total_duration_seconds,
                        "model": settings.model,
                        "final": final,
                    }
                )
            )
            return
        try:
            if chunk_path.stat().st_size <= 0:
                await websocket.send_bytes(
                    orjson.dumps(
                        {
                            "type": message_type,
                            "text": " ".join(part for part in transcript_parts if part).strip() if final else "",
                            "language": language,
                            "duration_seconds": total_duration_seconds,
                            "model": settings.model,
                            "final": final,
                        }
                    )
                )
                return
        except FileNotFoundError:
            await websocket.send_bytes(
                orjson.dumps(
                    {
                        "type": message_type,
                        "text": " ".join(part for part in transcript_parts if part).strip() if final else "",
                        "language": language,
                        "duration_seconds": total_duration_seconds,
                        "model": settings.model,
                        "final": final,
                    }
                )
            )
            chunk_path = None
            return
        incremental_prompt = _build_incremental_prompt(prompt, transcript_parts)
        try:
            result = await transcribe_path(chunk_path, language, incremental_prompt)
        except Exception as exc:
            await websocket.send_bytes(orjson.dumps({"type": "error", "detail": str(exc)}))
            return
        finally:
            await asyncio.to_thread(_delete_tmp_path, chunk_path)
            chunk_path = None
        chunk_text = result["text"].strip()
        if chunk_text:
            transcript_parts.append(chunk_text)
        total_duration_seconds += float(result["duration_seconds"])
        combined_text = " ".join(part for part in transcript_parts if part).strip()
        await websocket.send_bytes(
            orjson.dumps(
                {
                    "type": message_type,
                    "text": combined_text if final else chunk_text,
                    "language": result["language"],
                    "duration_seconds": total_duration_seconds,
                    "model": settings.model,
                    "final": final,
                }
            )
        )

    try:
        while True:
            if time.time() - started_at > settings.websocket_max_seconds:
                await websocket.close(code=1008, reason="stream exceeded maximum duration")
                return
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if data := message.get("bytes"):
                current_chunk = await ensure_chunk_path()
                await asyncio.to_thread(_append_bytes_to_path, current_chunk, data)
                if settings.websocket_auto_commit_bytes > 0:
                    chunk_size = await asyncio.to_thread(_path_size_bytes, current_chunk)
                    if chunk_size >= settings.websocket_auto_commit_bytes:
                        await flush_buffer(final=False)
                continue
            if text := message.get("text"):
                payload = json.loads(text)
                action = payload.get("type")
                if action == "start":
                    language = payload.get("language") or None
                    prompt = payload.get("prompt") or None
                    suffix = payload.get("suffix") or ".wav"
                    if chunk_path is not None:
                        await asyncio.to_thread(_delete_tmp_path, chunk_path)
                        chunk_path = None
                    transcript_parts = []
                    total_duration_seconds = 0.0
                    await websocket.send_bytes(orjson.dumps({"type": "ready", "model": settings.model}))
                elif action == "commit":
                    await flush_buffer(final=False)
                elif action == "finish":
                    await flush_buffer(final=True)
                    await websocket.close(code=1000)
                    return
                elif action == "reset":
                    if chunk_path is not None:
                        await asyncio.to_thread(_delete_tmp_path, chunk_path)
                        chunk_path = None
                    transcript_parts = []
                    total_duration_seconds = 0.0
                    await websocket.send_bytes(orjson.dumps({"type": "reset"}))
                else:
                    await websocket.send_bytes(orjson.dumps({"type": "error", "detail": f"unsupported action: {action}"}))
    except WebSocketDisconnect:
        return
    finally:
        if chunk_path is not None:
            await asyncio.to_thread(_delete_tmp_path, chunk_path)