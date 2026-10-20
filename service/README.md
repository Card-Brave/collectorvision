# CollectorVision service

A small HTTP service that identifies trading cards from photos using
[CollectorVision](../README.md): neural corner detection, dewarping, the Milo
embedder, and nearest-neighbour search over a Catalog v2 snapshot.

This directory is part of Card Brave's fork of
[HanClinto/CollectorVision](https://github.com/HanClinto/CollectorVision) and,
like the rest of the repository, is free software under **AGPL-3.0-or-later**.
Source: <https://github.com/Card-Brave/collectorvision>. Every response carries
an `X-Source-Code` header, and `GET /source` returns the same link (AGPL §13).

## What it adds over `examples/server`

- **Bearer-token auth** (`CV_TOKEN`) instead of open, wildcard-CORS endpoints.
- **Multi-frame consensus**: send several frames of the same card; frames that
  agree with the most confident one are averaged before search.
- **Orientation check**: each crop is also scored rotated 180°.
- **Ambiguity signal**: `margin` (top-1 minus top-2 score) and `ambiguous` flag
  near-ties — usually reprints sharing art — so callers can break them with set
  code / collector number OCR instead of trusting rank 1.
- **Blank-frame gate** on by default (`CV_MIN_SHARPNESS=0.02`).
- **Non-blocking inference**: endpoints run in worker threads with a bounded
  concurrency semaphore; models warm at startup.
- **Input limits**: frame count, bytes, pixel count, and downscaling of large photos.
- **Privacy**: images are never written to disk or logged; logs hold only result
  IDs, scores, and timing.
- **Offline images**: the Dockerfile bakes model and catalog snapshots in.

## API

`POST /v1/identify` (multipart, `Authorization: Bearer $CV_TOKEN`)

| Field | Description |
|---|---|
| `images` | One or more image files (JPEG/PNG) of the same card. Repeat the field per frame. |
| `game` | Optional loaded game, e.g. `mtg`. Omit to search all loaded catalogs. |
| `top_k` | Candidates to return (default 5, max 25). |
| `include_crop` | Return the dewarped crop as base64 JPEG. |
| `include_embedding` | Return the consensus query embedding. |

```bash
curl -s -H "Authorization: Bearer $CV_TOKEN" \
  -F images=@card-front.jpg -F images=@card-front-2.jpg -F game=mtg \
  http://localhost:8000/v1/identify
```

Response (abridged):

```json
{
  "card_present": true,
  "orientation": "upright",
  "frames_used": 2,
  "margin": 0.0051,
  "ambiguous": true,
  "candidates": [
    {"score": 0.827, "id": "d0def54b-…", "name": "Rush of Adrenaline",
     "identifiers": {"scryfall_card": "d0def54b-…", "scryfall_oracle": "…", "tcgplayer_product": "116450"},
     "set": "soi", "collector_number": "177", "lang": "en", "finishes": ["nonfoil", "foil"]}
  ],
  "frames": [{"card_present": true, "sharpness": 0.41, "used": true, "corners": [[…]]}],
  "timing": {"detect_embed_ms": 120.4, "total_ms": 126.9}
}
```

Other endpoints: `GET /healthz`, `GET /readyz`, `GET /source`, `GET /v1/info`
(authenticated: model and catalog versions).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CV_TOKEN` | — | Shared bearer token (required unless `CV_ALLOW_UNAUTHENTICATED=1`). |
| `CV_CATALOGS` | `mtg` | Comma-separated games, optionally `game:source` (e.g. `mtg:tcgplayer`). |

The Docker image bakes `mtg,pokemon,riftbound` by default; override with
`--build-arg CV_CATALOGS=...` (any game in `collector_vision/games.py`).
| `CV_PROVIDER` | `cpu` | ONNX Runtime provider: `cpu`, `gpu`, or `auto`. |
| `CV_THREADS` | `4` | ONNX Runtime threads per model. |
| `CV_MAX_CONCURRENCY` | `2` | Requests processed at once; others wait up to 30 s, then 503. |
| `CV_OFFLINE` | `0` | Never fetch catalogs/models (set in the Docker image). |
| `CV_MIN_SHARPNESS` | `0.02` | Detector presence gate; `0` disables. |
| `CV_FRAME_AGREEMENT` | `0.7` | Cosine similarity a frame needs with the anchor frame to join consensus. |
| `CV_AMBIGUITY_MARGIN` | `0.03` | `ambiguous` when top-1 minus top-2 is below this. |
| `CV_MAX_FRAMES` / `CV_MAX_IMAGE_BYTES` / `CV_MAX_SIDE` | `8` / 10 MiB / `1600` | Input limits. |
| `CV_SOURCE_URL` | this fork | Source link served for AGPL §13. Point it at your fork if you modify it. |

## Run

```bash
# Local
uv pip install -e ".[hf]" onnxruntime -e ./service
CV_ALLOW_UNAUTHENTICATED=1 python -m collectorvision_service

# Docker (from the repository root)
docker build -f service/Dockerfile -t collectorvision-service .
docker run -p 8000:8000 -e CV_TOKEN=change-me collectorvision-service
```

## Tests

```bash
cd service && python -m pytest tests          # includes real-model tests (downloads the MTG catalog)
cd service && python -m pytest tests -m "not hf"
```
