"""Run the service: ``python -m collectorvision_service``.

Modified by Card Brave (2026-10): new module. Licensed AGPL-3.0-or-later.
"""

import logging
import os

import uvicorn

from .app import create_app

logging.basicConfig(level=os.environ.get("CV_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
uvicorn.run(
    create_app(),
    host=os.environ.get("CV_HOST", "0.0.0.0"),
    port=int(os.environ.get("CV_PORT", "8000")),
    # Body size is bounded per image in the endpoint; keep a single worker per
    # container so catalogs and models load once. Scale with replicas instead.
    workers=1,
    access_log=False,
)
