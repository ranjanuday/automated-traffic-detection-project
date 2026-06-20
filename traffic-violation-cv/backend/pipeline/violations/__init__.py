"""Violation detector package.

Importing this package registers every built-in detector (side-effect of the
@register decorators). To add a new violation: drop a module here, decorate the
class, and import it below — the engine needs zero changes.
"""
from backend.pipeline.violations.base import all_detectors, register  # noqa: F401
from backend.pipeline.violations import riders   # noqa: F401
from backend.pipeline.violations import traffic  # noqa: F401
from backend.pipeline.violations import road     # noqa: F401

__all__ = ["all_detectors", "register"]
