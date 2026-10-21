"""Catalogs built from your own reference images, keyed by your own IDs.

Modified by Card Brave (2026-10): new module. Licensed AGPL-3.0-or-later.

A local catalog is one ``.npz`` file holding a normalized float16 embedding per
reference image plus a JSON record per row. Build one with::

    python -m collectorvision_service.local_catalog build \\
        --manifest references/manifest.jsonl --images references/images \\
        --name cardbrave --out catalogs/cardbrave.npz

Each manifest line describes one image (only ``id`` is required)::

    {"id": "printing-123", "image": "printing-123.jpg", "game": "azuki-tcg",
     "name": "Penny", "set": "S1", "set_name": "Genesis", "collector_number": "001",
     "lang": "en", "finishes": ["nonfoil"]}

Searches return ``id`` unchanged, so callers need no ID mapping. Rows carry their
own ``game``, so one file can hold several games and a request can be narrowed to
one of them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "collectorvision-service/local-catalog@1"
_IDENTITY = re.compile(r"(?P<model>[a-z0-9.-]+)@sha256:(?P<sha256>[0-9a-f]{64})$")


def model_key(identity: str) -> tuple[str, str]:
    """(model, sha256) of an embedding identity, ignoring who published the catalog."""
    match = _IDENTITY.search(identity)
    if match is None:
        raise ValueError(f"Unrecognized embedding model identity {identity!r}")
    return match.group("model"), match.group("sha256")


class LocalCatalog:
    """The subset of the CatalogV2 interface the service pipeline uses."""

    def __init__(
        self,
        *,
        name: str,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
        embedding_model: str,
        built_at: str,
    ) -> None:
        if embeddings.shape != (len(records), embeddings.shape[1]):
            raise ValueError("Local catalog embeddings and records differ in length")
        self.name = name
        self.embeddings = embeddings
        self.records = records
        self.embedding_model = embedding_model
        self.built_at = built_at
        self.source = f"local:{name}"
        self._games = np.array([record.get("game") or "" for record in records])

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> LocalCatalog:
        with np.load(Path(path), allow_pickle=False) as data:
            header = json.loads(str(data["header"]))
            if header.get("format") != FORMAT:
                raise ValueError(f"{path} is not a {FORMAT} file")
            embeddings = data["embeddings"].astype(np.float16, copy=False)
            records = [json.loads(line) for line in str(data["records"]).splitlines()]
        return cls(
            name=name or header["name"],
            embeddings=embeddings,
            records=records,
            embedding_model=header["embedding_model"],
            built_at=header.get("built_at", ""),
        )

    def games(self) -> list[str]:
        return sorted({game for game in self._games.tolist() if game})

    def search_records(
        self, embedding: np.ndarray, top_k: int = 5, game: str | None = None
    ) -> list[dict[str, Any]]:
        scores = self.embeddings @ embedding.astype(np.float16)
        scores = scores.astype(np.float32)
        if game is not None:
            scores = np.where(self._games == game, scores, -np.inf)
        count = min(top_k, len(scores))
        if count == 0:
            return []
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [self._record(int(index), float(scores[index])) for index in top if np.isfinite(scores[index])]

    def best_score(self, embedding: np.ndarray, game: str | None = None) -> float:
        hits = self.search_records(embedding, top_k=1, game=game)
        return hits[0]["score"] if hits else -1.0

    def _record(self, index: int, score: float) -> dict[str, Any]:
        record = self.records[index]
        return {
            "id": record["id"],
            "name": record.get("name"),
            "identifiers": {self.name: record["id"]},
            "face_index": 0,
            "finishes": record.get("finishes") or [],
            "metadata": {
                key: record.get(key)
                for key in ("game", "set", "set_name", "collector_number", "lang", "promo")
            },
            "score": score,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"LocalCatalog(name={self.name!r}, n={len(self)}, built_at={self.built_at!r})"


def build(
    manifest: Path,
    images: Path,
    out: Path,
    *,
    name: str,
    model: str = "milo",
    provider: str = "cpu",
    threads: int = 4,
) -> dict[str, Any]:
    """Embed every manifest image with the stable Milo model and write a catalog."""
    from PIL import Image

    from collector_vision.embedders.neural import NeuralEmbedder
    from collector_vision.model_artifacts import resolve_model_artifact
    from collector_vision.model_registry import get_model

    # The stable Milo release: the same model the published catalogs are built with,
    # so local and published catalogs can be searched with one query embedding.
    model = get_model(model)
    identity = f"local:{model.id}@sha256:{model.sha256}"
    embedder = NeuralEmbedder(
        checkpoint=resolve_model_artifact(model), num_threads=threads, provider=provider
    )

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    skipped: list[str] = []
    seen: set[str] = set()
    started = time.perf_counter()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row["id"])
        if identifier in seen:
            raise ValueError(f"Duplicate id {identifier!r} in {manifest}")
        seen.add(identifier)
        path = images / (row.get("image") or f"{identifier}.jpg")
        try:
            with Image.open(path) as image:
                vector = embedder.embed(image.convert("RGB"))
        except (OSError, ValueError) as error:
            skipped.append(f"{identifier}: {error}")
            continue
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        rows.append({key: value for key, value in row.items() if key != "image"} | {"id": identifier})
        if len(rows) % 250 == 0:
            print(f"embedded {len(rows)}", file=sys.stderr)

    if not rows:
        raise ValueError("No images could be embedded")
    header = {
        "format": FORMAT,
        "name": name,
        "embedding_model": identity,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(
        temporary,
        header=np.array(json.dumps(header)),
        embeddings=np.stack(vectors).astype(np.float16),
        records=np.array("\n".join(json.dumps(row, sort_keys=True) for row in rows)),
    )
    temporary.replace(out)
    return {
        **header,
        "skipped": skipped,
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(out),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a local CollectorVision catalog.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Embed reference images into a catalog")
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--images", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--name", required=True, help="Catalog name; also the identifier key")
    build_parser.add_argument("--model", default="milo", help="Registry model alias or exact ID")
    build_parser.add_argument("--provider", default="cpu")
    build_parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    result = build(
        args.manifest, args.images, args.out,
        name=args.name, model=args.model, provider=args.provider, threads=args.threads,
    )
    for problem in result.pop("skipped"):
        print(f"skipped {problem}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
