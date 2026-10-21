# Modified by Card Brave (2026-10): new file. Licensed AGPL-3.0-or-later.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from collectorvision_service.local_catalog import FORMAT, LocalCatalog, build, model_key

IMAGES = Path(__file__).resolve().parents[2] / "examples" / "images"
MILO = "milo-1.0.0@sha256:" + "a" * 64


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def catalog() -> LocalCatalog:
    vectors = np.stack([unit([1, 0, 0]), unit([0, 1, 0]), unit([0.9, 0.1, 0])]).astype(np.float16)
    records = [
        {"id": "p-1", "game": "azuki-tcg", "name": "Penny", "collector_number": "001"},
        {"id": "p-2", "game": "palworld", "name": "Lamball"},
        {"id": "p-3", "game": "palworld", "name": "Penny (reprint)"},
    ]
    return LocalCatalog(name="cardbrave", embeddings=vectors, records=records,
                        embedding_model=f"local:{MILO}", built_at="")


def test_model_key_ignores_publisher():
    published = "collectorvision@" + "0" * 40 + ":" + MILO
    assert model_key(published) == model_key(f"local:{MILO}") == ("milo-1.0.0", "a" * 64)
    with pytest.raises(ValueError):
        model_key("something-else")


def test_search_returns_own_ids_and_filters_by_game():
    local = catalog()
    hits = local.search_records(unit([1, 0, 0]), top_k=2)
    assert [hit["id"] for hit in hits] == ["p-1", "p-3"]
    assert hits[0]["identifiers"] == {"cardbrave": "p-1"}
    assert hits[0]["metadata"]["game"] == "azuki-tcg"
    assert hits[0]["metadata"]["collector_number"] == "001"

    palworld = local.search_records(unit([1, 0, 0]), top_k=5, game="palworld")
    assert [hit["id"] for hit in palworld] == ["p-3", "p-2"]
    assert local.search_records(unit([1, 0, 0]), game="unknown") == []
    assert local.games() == ["azuki-tcg", "palworld"]


def test_load_rejects_other_files(tmp_path: Path):
    path = tmp_path / "other.npz"
    np.savez(path, header=np.array(json.dumps({"format": "nope"})))
    with pytest.raises(ValueError):
        LocalCatalog.load(path)


@pytest.mark.hf
def test_build_and_load_round_trip(tmp_path: Path):
    images = sorted(IMAGES.glob("*.jpg"))[:2]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(
        json.dumps({"id": f"card-{index}", "image": path.name, "game": "mtg"})
        for index, path in enumerate(images)
    ) + "\n" + json.dumps({"id": "missing", "image": "missing.jpg"}) + "\n")
    out = tmp_path / "catalog.npz"
    result = build(manifest, IMAGES, out, name="cardbrave")
    assert result["count"] == 2 and len(result["skipped"]) == 1
    assert result["embedding_model"].startswith("local:milo-")

    loaded = LocalCatalog.load(out)
    assert json.loads(str(np.load(out)["header"]))["format"] == FORMAT
    assert len(loaded) == 2 and loaded.games() == ["mtg"]
    query = loaded.embeddings[1].astype(np.float32)
    assert loaded.search_records(query, top_k=1)[0]["id"] == "card-1"
