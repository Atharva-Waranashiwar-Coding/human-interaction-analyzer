import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import Annotation, Video
from app.services.video_service import parse_model_response, sample_video_frames

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    db.query(Annotation).delete()
    db.query(Video).delete()
    db.commit()
    db.close()
    os.environ.pop("MODEL_API_KEY", None)
    yield
    db = SessionLocal()
    db.query(Annotation).delete()
    db.query(Video).delete()
    db.commit()
    db.close()


def make_test_video(path: Path, duration_seconds: float = 3.0):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=10:duration={duration_seconds}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_upload_rejects_invalid_type():
    response = client.post("/api/videos", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_sample_video_frames_and_metadata_roundtrip():
    path = Path("/tmp/test-video-small.mp4")
    make_test_video(path, duration_seconds=3.0)
    try:
        frames = sample_video_frames(path, 3.0)
        assert frames
        for frame in frames:
            assert frame.exists()
    finally:
        if path.exists():
            path.unlink()


def test_model_response_parsing_and_annotation_validation():
    payload = '{"annotations": [{"start_time": 1.2, "end_time": 2.3, "behavior": "greeting", "participants": ["person_1", "person_2"], "description": "Both wave", "confidence": 0.9, "review_status": "needs_review"}]}'
    parsed = parse_model_response(payload)
    assert parsed[0]["behavior"] == "greeting"
    assert parsed[0]["participants"] == ["person_1", "person_2"]


def test_upload_and_manual_analyze_workflow(monkeypatch):
    monkeypatch.setattr("app.api.routes.run_model_inference", lambda *args: [])
    path = Path("/tmp/manual-mode-video.mp4")
    make_test_video(path, duration_seconds=2.0)
    try:
        with path.open("rb") as fh:
            response = client.post("/api/videos", files={"file": ("manual.mp4", fh.read(), "video/mp4")})
        assert response.status_code == 200, response.text
        video_id = response.json()["id"]
        analyze_response = client.post(f"/api/videos/{video_id}/analyze")
        assert analyze_response.status_code == 200, analyze_response.text
        assert analyze_response.json()["status"] == "processing"
        assert client.get(f"/api/videos/{video_id}").json()["analysis_status"] == "completed"

        annotation_response = client.put(
            f"/api/videos/{video_id}/annotations",
            json={
                "annotations": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "behavior": "greeting",
                        "participants": ["person_1", "person_2"],
                        "description": "Handshake",
                        "review_status": "reviewed",
                        "source": "manual",
                    }
                ]
            },
        )
        assert annotation_response.status_code == 200
        export_response = client.get(f"/api/videos/{video_id}/export?format=jsonl")
        assert export_response.status_code == 200
        assert export_response.text
    finally:
        if path.exists():
            path.unlink()


def test_duplicate_upload_prevention():
    path = Path("/tmp/duplicate-test.mp4")
    make_test_video(path, duration_seconds=2.0)
    try:
        with path.open("rb") as fh:
            first = client.post("/api/videos", files={"file": ("duplicate.mp4", fh.read(), "video/mp4")})
            fh.seek(0)
            second = client.post("/api/videos", files={"file": ("duplicate.mp4", fh.read(), "video/mp4")})
        assert first.status_code == 200
        assert second.status_code == 409
        assert "Duplicate video" in second.json()["detail"]
    finally:
        if path.exists():
            path.unlink()


def test_delete_video_cleans_storage():
    path = Path("/tmp/delete-test.mp4")
    make_test_video(path, duration_seconds=2.0)
    try:
        with path.open("rb") as fh:
            response = client.post("/api/videos", files={"file": ("delete.mp4", fh.read(), "video/mp4")})
        video_id = response.json()["id"]
        delete_response = client.delete(f"/api/videos/{video_id}")
        assert delete_response.status_code == 200
        check_response = client.get(f"/api/videos/{video_id}")
        assert check_response.status_code == 404
    finally:
        if path.exists():
            path.unlink()
