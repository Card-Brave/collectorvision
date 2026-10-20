# Modified by Card Brave (2026-10): new file. Licensed AGPL-3.0-or-later.
from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from collectorvision_service import Pipeline, PipelineError, Settings, create_app
from fastapi.testclient import TestClient

IMAGES = Path(__file__).resolve().parents[2] / "examples" / "images"


class StubPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def info(self) -> dict:
        return {"embedding_model": "stub", "catalogs": {}}

    def identify(self, images, **kwargs):
        self.calls.append({"images": images, **kwargs})
        if kwargs.get("game") == "nope":
            raise PipelineError("Unknown game 'nope'")
        return {
            "card_present": True,
            "frames_used": len(images),
            "candidates": [{"id": "abc", "score": 0.9}],
            "timing": {"total_ms": 1.0},
        }


def stub_client(**overrides) -> tuple[TestClient, StubPipeline]:
    stub = StubPipeline()
    settings = Settings(token="secret", **overrides)
    return TestClient(create_app(settings, pipeline=stub)), stub


def test_requires_token_configuration():
    with pytest.raises(RuntimeError):
        create_app(Settings(token=None), pipeline=StubPipeline())


def test_identify_requires_bearer_token():
    client, _ = stub_client()
    with client:
        files = [("images", ("a.jpg", b"x", "image/jpeg"))]
        assert client.post("/v1/identify", files=files).status_code == 401
        bad = client.post("/v1/identify", files=files, headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401
        ok = client.post("/v1/identify", files=files, headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        assert ok.json()["candidates"][0]["id"] == "abc"


def test_every_response_links_to_source():
    client, _ = stub_client()
    with client:
        for response in (client.get("/healthz"), client.get("/source"), client.get("/v1/info")):
            assert response.headers["X-Source-Code"] == "https://github.com/Card-Brave/collectorvision"
        assert client.get("/source").json()["license"] == "AGPL-3.0-or-later"


def test_forwards_frames_and_options_and_maps_errors():
    client, stub = stub_client()
    with client:
        headers = {"Authorization": "Bearer secret"}
        files = [("images", ("a.jpg", b"one", "image/jpeg")), ("images", ("b.jpg", b"two", "image/jpeg"))]
        response = client.post("/v1/identify", files=files, data={"game": "mtg", "top_k": "3"}, headers=headers)
        assert response.status_code == 200
        assert stub.calls[-1]["images"] == [b"one", b"two"]
        assert stub.calls[-1]["game"] == "mtg"
        assert stub.calls[-1]["top_k"] == 3
        failed = client.post("/v1/identify", files=files[:1], data={"game": "nope"}, headers=headers)
        assert failed.status_code == 422


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    try:
        loaded = Pipeline(Settings(token="secret"))
    except Exception as error:  # network or HF access unavailable
        pytest.skip(f"MTG catalog unavailable: {error}")
    loaded.warm()
    return loaded


def _jpeg(image) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.mark.hf
@pytest.mark.parametrize("path", sorted(IMAGES.glob("*.jpg")), ids=lambda path: path.name)
def test_identifies_sample_printings(pipeline: Pipeline, path: Path):
    expected = path.name.split("_", 1)[0]
    result = pipeline.identify([path.read_bytes()], game="mtg")
    assert result["card_present"]
    assert result["candidates"][0]["id"] == expected
    assert result["orientation"] == "upright"


@pytest.mark.hf
def test_upside_down_and_multi_frame(pipeline: Pipeline):
    path = sorted(IMAGES.glob("*_sample.jpg"))[0]
    expected = path.name.split("_", 1)[0]
    upright = cv2.imread(str(path))
    flipped = cv2.rotate(upright, cv2.ROTATE_180)

    single = pipeline.identify([_jpeg(flipped)], game="mtg")
    assert single["candidates"][0]["id"] == expected
    assert single["orientation"] in {"upright", "rotated_180"}

    combined = pipeline.identify([_jpeg(upright), _jpeg(flipped)], game="mtg")
    assert combined["candidates"][0]["id"] == expected
    assert combined["frames_used"] == 2


@pytest.mark.hf
def test_rejects_garbage_and_reports_no_card(pipeline: Pipeline):
    with pytest.raises(PipelineError):
        pipeline.identify([b"not an image"])
    blank = _jpeg(cv2.UMat(480, 640, cv2.CV_8UC3, (255, 255, 255)).get())
    result = pipeline.identify([blank], game="mtg")
    assert result["card_present"] is False
    assert result["candidates"] == []
