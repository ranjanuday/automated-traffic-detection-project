"""Shared dataclasses + geometry helpers used across the pipeline.

Keeping the domain vocabulary (Detection, Violation, Frame) in one tiny module
means every other component speaks the same language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class Detection:
    """A single detected object."""
    cls_name: str          # normalised category, e.g. "motorcycle", "person"
    confidence: float
    bbox: BBox
    raw_label: str = ""    # original model label

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class Violation:
    """A detected traffic violation with evidence pointers."""
    vtype: str             # one of config.VIOLATION_TYPES
    confidence: float
    bbox: BBox
    note: str = ""
    plate: Optional[str] = None


@dataclass
class Frame:
    """The unit of work flowing through the pipeline."""
    image: np.ndarray                      # working (possibly enhanced) image
    original: np.ndarray                   # untouched original
    detections: list[Detection] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    plates: dict[int, str] = field(default_factory=dict)  # det-index -> plate
    meta: dict = field(default_factory=dict)


# --- Geometry helpers ------------------------------------------------------
def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def contains_center(outer: BBox, inner: BBox) -> bool:
    """True if the centre of `inner` falls inside `outer`."""
    ox1, oy1, ox2, oy2 = outer
    cx = (inner[0] + inner[2]) // 2
    cy = (inner[1] + inner[3]) // 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def overlap_ratio(small: BBox, big: BBox) -> float:
    """Fraction of `small`'s area covered by `big` (association heuristic)."""
    sx1, sy1, sx2, sy2 = small
    bx1, by1, bx2, by2 = big
    ix1, iy1 = max(sx1, bx1), max(sy1, by1)
    ix2, iy2 = min(sx2, bx2), min(sy2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    sarea = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter / sarea


def union_bbox(boxes: list[BBox]) -> BBox:
    """Smallest box enclosing all given boxes."""
    xs1 = min(b[0] for b in boxes)
    ys1 = min(b[1] for b in boxes)
    xs2 = max(b[2] for b in boxes)
    ys2 = max(b[3] for b in boxes)
    return (xs1, ys1, xs2, ys2)
