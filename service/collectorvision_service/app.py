"""HTTP API for the CollectorVision service.

Modified by Card Brave (2026-10): new module, derived from the request flow of
``examples/server/server.py``. Licensed AGPL-3.0-or-later.

AGPL-3.0 §13: every response carries a link to this program's source code in the
``X-Source-Code`` header, and ``GET /source`` returns it.
"""

from __future__ import annotations

import hmac
import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import collector_vision as cvg

from .config import Settings
from .pipeline import Pipeline, PipelineError

logger = logging.getLogger("collectorvision_service")


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    """Build the app. Tests may inject a pipeline; production loads one at startup."""
    settings = settings or Settings.from_env()
    settings.validate()
    slots = threading.BoundedSemaphore(settings.max_concurrency)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = pipeline or Pipeline(settings)
        if settings.warm and pipeline is None:
            loaded.warm()
        app.state.pipeline = loaded
        logger.info("collectorvision service ready: %s", loaded.info())
        yield

    app = FastAPI(
        title="CollectorVision service",
        description=(
            "Card identification over HTTP. Free software under AGPL-3.0-or-later; "
            f"source: {settings.source_url}"
        ),
        version=cvg.__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def source_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Source-Code"] = settings.source_url
        return response

    def authorize(request: Request) -> None:
        if settings.allow_unauthenticated and not settings.token:
            return
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, settings.token or ""):
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        if getattr(request.app.state, "pipeline", None) is None:
            raise HTTPException(status_code=503, detail="Loading")
        return {"status": "ready"}

    @app.get("/source")
    def source():
        return {"license": "AGPL-3.0-or-later", "source": settings.source_url}

    @app.get("/v1/info", dependencies=[Depends(authorize)])
    def info(request: Request):
        return {
            "version": cvg.__version__,
            "license": "AGPL-3.0-or-later",
            "source": settings.source_url,
            **request.app.state.pipeline.info(),
        }

    # Sync endpoint: FastAPI runs it in a worker thread, so CPU-bound inference
    # does not block the event loop. The semaphore bounds memory under load.
    @app.post("/v1/identify", dependencies=[Depends(authorize)])
    def identify(
        request: Request,
        images: Annotated[list[UploadFile], File(description="1+ frames of the same card")],
        game: Annotated[str | None, Form()] = None,
        top_k: Annotated[int | None, Form()] = None,
        include_crop: Annotated[bool, Form()] = False,
        include_embedding: Annotated[bool, Form()] = False,
    ):
        if not slots.acquire(timeout=30):
            raise HTTPException(status_code=503, detail="Recognition is busy; retry shortly")
        try:
            data = [upload.file.read(settings.max_image_bytes + 1) for upload in images]
            result = request.app.state.pipeline.identify(
                data,
                game=game or None,
                top_k=top_k,
                include_crop=include_crop,
                include_embedding=include_embedding,
            )
        except PipelineError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            slots.release()
        best = result["candidates"][0] if result["candidates"] else None
        # Log identities and scores only — never images or client data.
        logger.info(
            "identify frames=%d used=%s card=%s score=%s ambiguous=%s ms=%s",
            len(images),
            result.get("frames_used"),
            best and best["id"],
            best and best["score"],
            result.get("ambiguous"),
            result["timing"]["total_ms"],
        )
        return JSONResponse(result)

    return app
