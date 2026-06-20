"""Violation detector base class + registry.

Every violation type is a self-contained detector implementing `run(ctx)`.
Register it with @register and the engine picks it up automatically — no
central switch statement to edit (Open/Closed principle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.pipeline.types import Detection, Frame, Violation

_REGISTRY: list["ViolationDetector"] = []


def register(cls):
    """Class decorator: instantiate + add to the active detector list."""
    _REGISTRY.append(cls())
    return cls


def all_detectors() -> list["ViolationDetector"]:
    return list(_REGISTRY)


class ViolationDetector(ABC):
    """Contract every violation detector must honour."""

    vtype: str = "unknown"

    @abstractmethod
    def run(self, frame: Frame) -> list[Violation]:
        """Inspect the frame and return zero or more violations."""
        raise NotImplementedError

    # --- shared helpers available to all detectors ------------------------
    @staticmethod
    def vehicles(frame: Frame) -> list[Detection]:
        from backend import config
        return [d for d in frame.detections
                if d.cls_name in config.VEHICLE_CATEGORIES]

    @staticmethod
    def persons(frame: Frame) -> list[Detection]:
        return [d for d in frame.detections if d.cls_name == "person"]

    @staticmethod
    def of(frame: Frame, cls_name: str) -> list[Detection]:
        return [d for d in frame.detections if d.cls_name == cls_name]
