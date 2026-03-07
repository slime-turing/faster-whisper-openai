import json
import os
import unittest

os.environ["STT_SKIP_MODEL_LOAD"] = "1"
os.environ.setdefault("STT_CACHE_DIR", "/tmp/faster-whisper-smoke-cache")
os.environ.setdefault("STT_TMP_DIR", "/tmp/faster-whisper-smoke-tmp")

from fastapi.testclient import TestClient

from app import main as main_module


async def fake_transcribe_path(path, language, prompt):
    return {
        "text": "smoke transcript",
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

            websocket.send_bytes(b"fake-audio")
            websocket.send_text(json.dumps({"type": "commit"}))
            partial = json.loads(websocket.receive_bytes())
            self.assertEqual(partial["type"], "partial")
            self.assertEqual(partial["text"], "smoke transcript")
            self.assertFalse(partial["final"])

            websocket.send_text(json.dumps({"type": "finish"}))
            final = json.loads(websocket.receive_bytes())
            self.assertEqual(final["type"], "final")
            self.assertTrue(final["final"])


if __name__ == "__main__":
    unittest.main()