"""Multi-frame, orientation-aware identification on top of CollectorVision.

Modified by Card Brave (2026-10): new module. Licensed AGPL-3.0-or-later like the
rest of this repository.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import collector_vision as cvg
from collector_vision import retrieval
from collector_vision.embedders.neural import NeuralEmbedder
from collector_vision.model_artifacts import resolve_model_artifact
from collector_vision.model_registry import get_model

from .config import Settings
from .local_catalog import LocalCatalog, model_key


logger = logging.getLogger("collectorvision_service")


class PipelineError(ValueError):
    """A request the pipeline cannot process (bad image, unknown game, …)."""


@dataclass
class FrameResult:
    card_present: bool
    sharpness: float | None = None
    orientation: str | None = None  # "upright" | "rotated_180"
    top_score: float | None = None
    used: bool = False
    corners: list[list[float]] | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)
    crop: Any = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "card_present": self.card_present,
            "sharpness": None if self.sharpness is None else round(float(self.sharpness), 5),
            "orientation": self.orientation,
            "top_score": None if self.top_score is None else round(self.top_score, 4),
            "used": self.used,
            "corners": self.corners,
        }


def _parse_catalog_spec(raw: str) -> tuple[str, str | None]:
    game, _, source = raw.partition(":")
    return game.strip(), (source.strip() or None)


# A search target: a loaded catalog plus, for local catalogs holding several games,
# the one game to restrict results to.
Target = tuple[str, Any, str | None]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class Pipeline:
    """Loads catalogs and models once; identifies cards from one or more frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Published catalogs ("mtg", "pokemon:tcgplayer") and local catalogs built
        # from your own images ("cardbrave=/catalogs/cardbrave.npz").
        self.catalogs: dict[str, Any] = {}
        for raw in settings.catalogs:
            name, _, path = raw.partition("=")
            if path:
                # A local catalog is built after first deploy, so a missing file is
                # logged and skipped rather than keeping the published catalogs down.
                if not Path(path.strip()).is_file():
                    logger.warning("local catalog %s not found at %s; skipping", name.strip(), path.strip())
                    continue
                self.catalogs[name.strip()] = LocalCatalog.load(path.strip(), name=name.strip())
                continue
            game, source = _parse_catalog_spec(raw)
            self.catalogs[game] = cvg.CatalogV2.load(
                game,
                source=source,
                cache_dir=settings.cache_dir,
                offline=settings.offline,
            )
        if not self.catalogs:
            raise RuntimeError("No catalogs configured (CV_CATALOGS).")

        # Every catalog must come from the same model, or one query embedding cannot
        # search them all. Published identities also name the publishing commit.
        models = {model_key(catalog.embedding_model) for catalog in self.catalogs.values()}
        if len(models) != 1:
            raise RuntimeError(f"Loaded catalogs use different embedding models: {sorted(models)}")
        self.embedding_model = next(iter(self.catalogs.values())).embedding_model
        self.embedder = self._build_embedder(self.embedding_model)
        self.detector = cvg.NeuralCornerDetector(
            num_threads=settings.threads, provider=settings.provider
        )

    def _build_embedder(self, identity: str) -> NeuralEmbedder:
        # Same resolution as CatalogV2.embedder, but honouring the configured provider
        # (CoreML/accelerator sessions can fail at runtime on some hosts).
        name, sha256 = model_key(identity)
        model = get_model(name)
        if model.sha256 != sha256:
            raise RuntimeError(f"Installed model registry does not match {identity!r}")
        checkpoint = resolve_model_artifact(model, cache_dir=self.settings.cache_dir)
        return NeuralEmbedder(
            checkpoint=checkpoint, num_threads=self.settings.threads, provider=self.settings.provider
        )

    def warm(self) -> None:
        """Run each model once so the first real request is not slow."""
        blank = np.full((680, 488, 3), 127, dtype=np.uint8)
        self.detector.detect(blank, min_sharpness=0.0)
        self.embedder.embed(cvg.DetectionResult(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        ).dewarp(blank))

    def info(self) -> dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "catalogs": {
                game: {
                    "source": catalog.source,
                    "size": len(catalog),
                    "catalog": repr(catalog),
                    **({"games": catalog.games()} if isinstance(catalog, LocalCatalog) else {}),
                }
                for game, catalog in self.catalogs.items()
            },
        }

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def games(self) -> list[str]:
        names = set(self.catalogs)
        for catalog in self.catalogs.values():
            if isinstance(catalog, LocalCatalog):
                names.update(catalog.games())
        return sorted(names)

    def _targets(self, game: str | None) -> list[Target]:
        if game is None:
            return [(name, catalog, None) for name, catalog in self.catalogs.items()]
        if game in self.catalogs:
            return [(game, self.catalogs[game], None)]
        targets = [
            (name, catalog, game)
            for name, catalog in self.catalogs.items()
            if isinstance(catalog, LocalCatalog) and game in catalog.games()
        ]
        if not targets:
            raise PipelineError(f"Unknown game {game!r}; loaded games: {self.games()}")
        return targets

    @staticmethod
    def _search(target: Target, vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        _, catalog, game = target
        if isinstance(catalog, LocalCatalog):
            return catalog.search_records(vector, top_k=top_k, game=game)
        return catalog.search_records(vector, top_k=top_k)

    def _best_score(self, targets: list[Target], vector: np.ndarray) -> float:
        scores = []
        for target in targets:
            catalog = target[1]
            if isinstance(catalog, LocalCatalog):
                scores.append(catalog.best_score(vector, game=target[2]))
            else:
                scores.append(retrieval.cosine_search(vector, catalog.embeddings, top_k=1)[0][0])
        return max(scores)

    def decode(self, data: bytes) -> np.ndarray:
        settings = self.settings
        if len(data) > settings.max_image_bytes:
            raise PipelineError(f"Image exceeds {settings.max_image_bytes} bytes")
        raw = np.frombuffer(data, dtype=np.uint8)
        # A 1/8-scale decode is cheap for JPEG and bounds full-size allocation for
        # oversized (or decompression-bomb) inputs before the real decode.
        header = cv2.imdecode(raw, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if header is None:
            raise PipelineError("Could not decode image (unsupported format or corrupt data)")
        if header.shape[0] * header.shape[1] * 64 > settings.max_pixels:
            raise PipelineError(f"Image exceeds {settings.max_pixels} pixels")
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise PipelineError("Could not decode image (unsupported format or corrupt data)")
        height, width = bgr.shape[:2]
        scale = settings.max_side / max(height, width)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        return bgr

    def _frame(self, bgr: np.ndarray, catalogs: list[Target]) -> FrameResult:
        detection = self.detector.detect(bgr, min_sharpness=self.settings.min_sharpness)
        if not detection.card_present or detection.corners is None:
            return FrameResult(card_present=False, sharpness=detection.sharpness)
        crop = detection.dewarp(bgr)
        # The corner detector usually returns an upright crop even for rotated photos,
        # but Milo embeddings are sensitive to 180° rotation when it does not, so score
        # both orientations and keep the stronger one (one extra embedding per frame).
        embeddings = self.embedder.embed([crop, cvg.rotate_card_180(crop)])
        scores = [self._best_score(catalogs, embeddings[index]) for index in range(2)]
        index = int(np.argmax(scores))
        return FrameResult(
            card_present=True,
            sharpness=detection.sharpness,
            orientation="upright" if index == 0 else "rotated_180",
            top_score=float(scores[index]),
            corners=np.asarray(detection.corners, dtype=float).round(5).tolist(),
            embedding=embeddings[index],
            crop=crop if index == 0 else cvg.rotate_card_180(crop),
        )

    def identify(
        self,
        images: list[bytes],
        *,
        game: str | None = None,
        top_k: int | None = None,
        include_crop: bool = False,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        settings = self.settings
        started = time.perf_counter()
        if not images:
            raise PipelineError("At least one image is required")
        if len(images) > settings.max_frames:
            raise PipelineError(f"At most {settings.max_frames} frames per request")
        top_k = top_k or settings.top_k
        if not 1 <= top_k <= settings.max_top_k:
            raise PipelineError(f"top_k must be between 1 and {settings.max_top_k}")
        catalogs = self._targets(game)

        frames = [self._frame(self.decode(data), catalogs) for data in images]
        present = [frame for frame in frames if frame.card_present]
        timing = {"detect_embed_ms": round((time.perf_counter() - started) * 1000, 1)}
        if not present:
            return {
                "card_present": False,
                "frames": [frame.public() for frame in frames],
                "candidates": [],
                "timing": {**timing, "total_ms": timing["detect_embed_ms"]},
            }

        # Consensus: anchor on the most confident frame and average every frame that
        # agrees with it. Bad corner grabs land far away and are excluded.
        anchor = max(present, key=lambda frame: frame.top_score or 0.0)
        agreeing = [
            frame for frame in present
            if float(np.dot(anchor.embedding, frame.embedding)) >= settings.frame_agreement
        ]
        for frame in agreeing:
            frame.used = True
        query = _normalize(np.stack([frame.embedding for frame in agreeing]).mean(axis=0))

        hits: list[dict[str, Any]] = []
        for target in catalogs:
            name, catalog, _ = target
            for record in self._search(target, query, top_k):
                hits.append(_candidate(name, catalog.source, record))
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        hits = hits[:top_k]
        margin = hits[0]["score"] - hits[1]["score"] if len(hits) > 1 else hits[0]["score"]

        result: dict[str, Any] = {
            "card_present": True,
            "orientation": anchor.orientation,
            "frames": [frame.public() for frame in frames],
            "frames_used": len(agreeing),
            "candidates": hits,
            "margin": round(margin, 4),
            # Near-ties are usually reprints sharing art; the caller should break
            # them with set code / collector number OCR rather than trust rank 1.
            "ambiguous": margin < settings.ambiguity_margin,
            "timing": {**timing, "total_ms": round((time.perf_counter() - started) * 1000, 1)},
        }
        if include_embedding:
            result["embedding"] = [round(float(value), 6) for value in query]
        if include_crop:
            result["crop_jpeg"] = _jpeg_base64(anchor.crop)
        return result


def _candidate(game: str, source: str, record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "score": round(float(record["score"]), 4),
        "id": record["id"],
        # Local catalogs can hold several games; each row names its own.
        "game": metadata.get("game") or game,
        "catalog": game,
        "source": source,
        "name": record.get("name"),
        "identifiers": record.get("identifiers") or {},
        "face_index": record.get("face_index", 0),
        "finishes": record.get("finishes") or [],
        "set": metadata.get("set"),
        "set_name": metadata.get("set_name"),
        "collector_number": metadata.get("collector_number"),
        "lang": metadata.get("lang"),
        "promo": metadata.get("promo"),
    }


def _jpeg_base64(crop: Any, max_side: int = 400) -> str:
    bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise PipelineError("Could not encode crop")
    return base64.b64encode(buffer.tobytes()).decode()
