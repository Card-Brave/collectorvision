"""Environment-driven settings for the CollectorVision service.

Modified by Card Brave (2026-10): new module. Licensed AGPL-3.0-or-later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE_URL = "https://github.com/Card-Brave/collectorvision"


def _bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else int(value)


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else float(value)


@dataclass(frozen=True)
class Settings:
    catalogs: tuple[str, ...] = ("mtg",)
    token: str | None = None
    allow_unauthenticated: bool = False
    provider: str = "cpu"
    threads: int = 4
    max_concurrency: int = 2
    offline: bool = False
    cache_dir: Path | None = None
    top_k: int = 5
    max_top_k: int = 25
    max_frames: int = 8
    max_image_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 40_000_000
    max_side: int = 1600
    # Library default. 0 disables the gate and reports a "card" on blank frames.
    min_sharpness: float = 0.02
    frame_agreement: float = 0.7
    ambiguity_margin: float = 0.03
    source_url: str = DEFAULT_SOURCE_URL
    warm: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        catalogs = tuple(
            part.strip() for part in os.environ.get("CV_CATALOGS", "mtg").split(",") if part.strip()
        )
        cache = os.environ.get("CV_CACHE_DIR") or os.environ.get("COLLECTORVISION_CACHE")
        return cls(
            catalogs=catalogs,
            token=os.environ.get("CV_TOKEN") or None,
            allow_unauthenticated=_bool("CV_ALLOW_UNAUTHENTICATED"),
            provider=os.environ.get("CV_PROVIDER", "cpu"),
            threads=_int("CV_THREADS", 4),
            max_concurrency=_int("CV_MAX_CONCURRENCY", 2),
            offline=_bool("CV_OFFLINE"),
            cache_dir=Path(cache).expanduser() if cache else None,
            top_k=_int("CV_TOP_K", 5),
            max_frames=_int("CV_MAX_FRAMES", 8),
            max_image_bytes=_int("CV_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
            max_side=_int("CV_MAX_SIDE", 1600),
            min_sharpness=_float("CV_MIN_SHARPNESS", 0.02),
            frame_agreement=_float("CV_FRAME_AGREEMENT", 0.7),
            ambiguity_margin=_float("CV_AMBIGUITY_MARGIN", 0.03),
            source_url=os.environ.get("CV_SOURCE_URL", DEFAULT_SOURCE_URL),
            warm=_bool("CV_WARM", True),
        )

    def validate(self) -> None:
        if not self.token and not self.allow_unauthenticated:
            raise RuntimeError(
                "Set CV_TOKEN (shared bearer token), or CV_ALLOW_UNAUTHENTICATED=1 for local use."
            )
        if self.provider not in {"auto", "cpu", "gpu"}:
            raise RuntimeError("CV_PROVIDER must be auto, cpu, or gpu")
