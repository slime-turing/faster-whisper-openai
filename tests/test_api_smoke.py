import json
import os
import unittest
from dataclasses import replace
from unittest import mock

os.environ["STT_SKIP_MODEL_LOAD"] = "1"
os.environ.setdefault("STT_CACHE_DIR", "/tmp/faster-whisper-smoke-cache")
os.environ.setdefault("STT_TMP_DIR", "/tmp/faster-whisper-smoke-tmp")

from fastapi.testclient import TestClient

from app import main as main_module


async def fake_transcribe_path(path, language, prompt):
    raw = path.read_bytes()
    decoded = raw.decode("utf-8") if raw and raw != b"fake-audio" else "smoke transcript"
    return {
        "text": decoded,
        "language": language or "en",
        "duration_seconds": 1.25,
    }


class ApiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.original_transcribe_path = main_module.transcribe_path
        main_module.transcribe_path = fake_transcribe_path
        self.client = TestClient(main_module.app)

    def tearDown(self):
        main_module.transcribe_path = self.original_transcribe_path
        self.client.close()

    def test_health_endpoint(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model", payload)

    def test_health_endpoint_uses_cached_storage_metrics(self):
        original_snapshot = main_module._snapshot_storage_metrics
        original_dir_size = main_module._dir_size_bytes

        try:
            main_module._snapshot_storage_metrics = lambda: main_module.StorageMetrics(
                cache_size_bytes=11,
                tmp_size_bytes=7,
            )

            def fail_dir_size(_root):
                raise AssertionError("healthz should use cached storage metrics")

            main_module._dir_size_bytes = fail_dir_size
            response = self.client.get("/healthz")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["cache_size_bytes"], 18)
        finally:
            main_module._snapshot_storage_metrics = original_snapshot
            main_module._dir_size_bytes = original_dir_size

    def test_health_endpoint_tolerates_disappearing_files(self):
        cache_root = main_module.settings.cache_dir
        stable_file = cache_root / "stable.bin"
        vanishing_file = cache_root / "vanishing.bin"
        stable_file.write_bytes(b"1234")
        vanishing_file.write_bytes(b"5678")

        original_rglob = main_module.Path.rglob

        def flaky_rglob(path_obj, pattern):
            for item in original_rglob(path_obj, pattern):
                if item == vanishing_file and item.exists():
                    item.unlink()
                yield item

        try:
            with mock.patch.object(main_module.Path, "rglob", autospec=True, side_effect=flaky_rglob):
                response = self.client.get("/healthz")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")
        finally:
            stable_file.unlink(missing_ok=True)
            vanishing_file.unlink(missing_ok=True)

    def test_cleanup_skips_active_tmp_file_until_unmarked(self):
        tmp_root = main_module.settings.tmp_dir
        active_file = tmp_root / "active.wav"
        active_file.write_bytes(b"busy")

        try:
            main_module._mark_tmp_file_active(active_file)
            future_now = active_file.stat().st_mtime + (main_module.settings.tmp_ttl_hours * 3600) + 1
            main_module._cleanup_tmp_files(future_now)
            self.assertTrue(active_file.exists())

            main_module._unmark_tmp_file_active(active_file)
            main_module._cleanup_tmp_files(future_now)
            self.assertFalse(active_file.exists())
        finally:
            main_module._unmark_tmp_file_active(active_file)
            active_file.unlink(missing_ok=True)

    def test_openapi_contains_expected_paths(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["info"]["title"], "Faster Whisper OpenAI Wrapper")
        self.assertIn("/v1/audio/transcriptions", payload["paths"])
        self.assertIn("/healthz", payload["paths"])

    def test_http_transcription_json_response(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            data={"model": "large-v3-turbo"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": "smoke transcript",
                "model": main_module.settings.model,
                "language": "en",
                "duration_seconds": 1.25,
            },
        )

    def test_http_transcription_accepts_whisper_alias(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            data={"model": "whisper-1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], main_module.settings.model)

    def test_http_transcription_text_response(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", b"fake-audio", "audio/wav")},
            data={"model": "large-v3-turbo", "response_format": "text"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "smoke transcript")

    def test_websocket_contract(self):
        with self.client.websocket_connect("/v1/realtime/transcriptions") as websocket:
            websocket.send_text(json.dumps({"type": "start", "language": "en", "suffix": ".wav"}))
            ready = json.loads(websocket.receive_bytes())
            self.assertEqual(ready["type"], "ready")

            websocket.send_bytes(b"chunk-one")
            websocket.send_text(json.dumps({"type": "commit"}))
            partial = json.loads(websocket.receive_bytes())
            self.assertEqual(partial["type"], "partial")
            self.assertEqual(partial["text"], "chunk-one")
            self.assertFalse(partial["final"])

            websocket.send_bytes(b"chunk-two")
            websocket.send_text(json.dumps({"type": "finish"}))
            final = json.loads(websocket.receive_bytes())
            self.assertEqual(final["type"], "final")
            self.assertTrue(final["final"])
            self.assertEqual(final["text"], "chunk-one chunk-two")

    def test_websocket_does_not_leave_stream_temp_files_after_finish(self):
        tmp_root = main_module.settings.tmp_dir
        before = {path.name for path in tmp_root.glob("stream-*")}

        with self.client.websocket_connect("/v1/realtime/transcriptions") as websocket:
            websocket.send_text(json.dumps({"type": "start", "language": "en", "suffix": ".wav"}))
            json.loads(websocket.receive_bytes())
            websocket.send_bytes(b"fake-audio")
            websocket.send_text(json.dumps({"type": "finish"}))
            json.loads(websocket.receive_bytes())

        after = {path.name for path in tmp_root.glob("stream-*")}
        self.assertEqual(after, before)

    def test_websocket_auto_commits_when_chunk_reaches_threshold(self):
        original_settings = main_module.settings
        main_module.settings = replace(main_module.settings, websocket_auto_commit_bytes=4)
        try:
            with self.client.websocket_connect("/v1/realtime/transcriptions") as websocket:
                websocket.send_text(json.dumps({"type": "start", "language": "en", "suffix": ".wav"}))
                ready = json.loads(websocket.receive_bytes())
                self.assertEqual(ready["type"], "ready")

                websocket.send_bytes(b"auto")
                partial = json.loads(websocket.receive_bytes())
                self.assertEqual(partial["type"], "partial")
                self.assertEqual(partial["text"], "auto")
                self.assertFalse(partial["final"])

                websocket.send_text(json.dumps({"type": "finish"}))
                final = json.loads(websocket.receive_bytes())
                self.assertEqual(final["type"], "final")
                self.assertEqual(final["text"], "auto")
        finally:
            main_module.settings = original_settings


if __name__ == "__main__":
    unittest.main()