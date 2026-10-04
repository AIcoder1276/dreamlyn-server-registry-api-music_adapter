import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TMP_DIR = tempfile.mkdtemp()
CONFIG_PATH = Path(TMP_DIR) / "config.yml"
CONFIG_PATH.write_text(
    yaml.safe_dump(
        {
            "musicgen_api_base": "http://musicgen.invalid:8000",
            "s3": {
                "endpoint": "http://swfs-s3:8333",
                "bucket": "dreamlyn-music",
                "prefix": "musicgen",
                "public_base_url": "http://s3.example.test",
            },
            "folders": {"logs": str(Path(TMP_DIR) / "logs")},
            "generation": {
                "duration_seconds": {"min": 5, "max": 7, "default": 6},
                "default_format": "wav",
            },
        }
    ),
    encoding="utf-8",
)
os.environ.setdefault("MUSICAL_CONFIG_PATH", str(CONFIG_PATH))
os.environ.setdefault("STORAGE_MUSIC_USER_LOGIN", "test-user")
os.environ.setdefault("STORAGE_MUSIC_USER_PASSWORD", "test-password")

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class AcceptedModelsTests(unittest.TestCase):
    def setUp(self):
        self.original_models = set(app.ACCEPTED_MODELS)
        patches = [
            mock.patch.object(app, "request_musicgen", mock.AsyncMock(return_value=b"RIFF")),
            mock.patch.object(app, "upload_audio", return_value="http://s3.example.test/key.wav"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        # No lifespan run: only the static names are known, as when MusicGen boots after the adapter.
        self.client = TestClient(app.app)

    def tearDown(self):
        app.ACCEPTED_MODELS.clear()
        app.ACCEPTED_MODELS.update(self.original_models)

    def post(self, model: str):
        return self.client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "calm piano"}]},
        )

    def test_litellm_model_name_is_accepted_without_inference(self):
        response = self.post("Facebook/MusicGen-Small")

        self.assertEqual(200, response.status_code, response.text)

    def test_unknown_model_refreshes_names_from_inference(self):
        async def resolve():
            app.ACCEPTED_MODELS.add("facebook/musicgen-small")

        with mock.patch.object(app, "refresh_accepted_models", side_effect=resolve) as refresh:
            response = self.post("facebook/musicgen-small")

        refresh.assert_awaited_once()
        self.assertEqual(200, response.status_code, response.text)

    def test_unknown_model_is_rejected_after_refresh(self):
        with mock.patch.object(app, "refresh_accepted_models", mock.AsyncMock()) as refresh:
            response = self.post("not-a-model")

        refresh.assert_awaited_once()
        self.assertEqual(400, response.status_code)


if __name__ == "__main__":
    unittest.main()
