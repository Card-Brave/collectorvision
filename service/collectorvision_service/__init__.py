"""Card identification HTTP service built on CollectorVision.

Modified by Card Brave (2026-10): new package. Licensed AGPL-3.0-or-later.
"""

from .app import create_app
from .config import Settings
from .pipeline import Pipeline, PipelineError

__all__ = ["Pipeline", "PipelineError", "Settings", "create_app"]
