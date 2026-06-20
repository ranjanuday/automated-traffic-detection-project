"""Signal/line violations: red-light jumping & stop-line crossing.

These need scene context (where the stop line is, signal state). We infer the
signal colour from the detected traffic-light crop via HSV, and accept an
optional per-image stop-line as a normalised y in frame.meta["stop_line_y"]
(0-1). If absent we default to 0.55 of image height — configurable per camera
in a real deployment.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.pipeline.types import Detection, Frame, Violation
from backend.pipeline.violations.base import ViolationDetector, register


def _signal_state(frame: Frame, light: Detection) -> str:
    """Classify a traffic light crop as red/amber/green/unknown via HSV."""
    x1, y1, x2, y2 = light.bbox
    crop = frame.image[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "red": (cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) +
                cv2.inRange(hsv, (170, 90, 90), (180, 255, 255))),
        "amber": cv2.inRange(hsv, (15, 90, 90), (35, 255, 255)),
        "green": cv2.inRange(hsv, (40, 60, 60), (90, 255, 255)),
    }
    scores = {k: int(np.count_nonzero(m)) for k, m in masks.items()}
    total = max(1, crop.shape[0] * crop.shape[1])
    best = max(scores, key=scores.get)
    runner_up = max((v for k, v in scores.items() if k != best), default=0)
    # A lit lamp occupies a real fraction of the housing AND clearly dominates
    # the other colours. This rejects dark/unlit housings and stray red/green
    # pixels bleeding in from the background (e.g. a red car behind the pole).
    if scores[best] < 0.04 * total:
        return "unknown"
    if scores[best] < 1.5 * runner_up:
        return "unknown"
    return best


def _detect_white_line(image: np.ndarray) -> int | None:
    """Find a prominent horizontal white road marking (stop line) -> y or None.

    Scans the road area (lower ~55% of the frame) for a near-horizontal band of
    bright pixels spanning a good fraction of the width. Returns its y, or None
    when no clear marking is present (so we don't invent a stop line).
    """
    h, w = image.shape[:2]
    roi_top = int(h * 0.55)  # stop lines sit low in the frame
    gray = cv2.cvtColor(image[roi_top:], cv2.COLOR_BGR2GRAY)
    # Bright road markings only.
    _, white = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
    # Per-row coverage; a stop line spans a large fraction of the road width.
    row_cov = white.sum(axis=1) / 255.0
    need = 0.55 * w  # must span most of the carriageway, not a stray bright spot
    rows = np.where(row_cov > need)[0]
    if rows.size == 0:
        return None
    # A real stop line is a THIN, CONTIGUOUS band. Find the largest contiguous
    # run of qualifying rows; reject diffuse bright regions (sunlit road, sky,
    # walls) which produce many scattered/thick bright rows.
    splits = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    band = max(splits, key=len)
    band_thick = band.size
    if band_thick > 0.10 * white.shape[0]:   # too thick -> not a line
        return None
    if rows.size > 1.8 * band_thick:          # too much scattered brightness
        return None
    best = band[int(np.argmax(row_cov[band]))]
    return roi_top + int(best)


def _find_stop_line(frame: Frame) -> tuple[int | None, str | None]:
    """Resolve the stop line: configured (per-camera) > detected > unknown."""
    if "stop_line_y" in frame.meta:
        norm = float(frame.meta["stop_line_y"])
        return int(frame.image.shape[0] * norm), "configured"
    detected = _detect_white_line(frame.image)
    if detected is not None:
        return detected, "detected"
    return None, None


@register
class RedLightDetector(ViolationDetector):
    """Vehicle past the stop line while the signal is red."""

    vtype = "red_light"

    def run(self, frame: Frame) -> list[Violation]:
        lights = self.of(frame, "traffic_light")
        if not lights:
            return []
        states = [_signal_state(frame, lt) for lt in lights]
        if "red" not in states:
            return []
        line_y, _src = _find_stop_line(frame)
        if line_y is None:
            return []  # can't judge crossing without a stop line
        out: list[Violation] = []
        for v in self.vehicles(frame):
            # Front of the vehicle (bottom edge) is past the line.
            if v.bbox[3] > line_y and v.bbox[1] < line_y:
                out.append(Violation(
                    vtype=self.vtype, confidence=0.75, bbox=v.bbox,
                    note=f"{v.cls_name} crossed stop line on RED signal",
                ))
        return out


@register
class StopLineDetector(ViolationDetector):
    """Vehicle clearly over a KNOWN stop line (configured or detected).

    Critically, this only fires when an actual stop line is established - either
    supplied per-camera via meta["stop_line_y"] or auto-detected as a white road
    marking. Without a real line we abstain (no false positives).
    """

    vtype = "stop_line"

    def run(self, frame: Frame) -> list[Violation]:
        line_y, src = _find_stop_line(frame)
        if line_y is None:
            return []  # no stop line known -> nothing to judge
        out: list[Violation] = []
        for v in self.vehicles(frame):
            x1, y1, x2, y2 = v.bbox
            height = max(1, y2 - y1)
            # The line must cut the LOWER part of the vehicle: its front/nose
            # has clearly overrun the line (not merely a tall box spanning it).
            crossed = (y1 < line_y < y2) and (line_y - y1) > 0.55 * height
            if crossed:
                conf = 0.6 if src == "detected" else 0.65
                out.append(Violation(
                    vtype=self.vtype, confidence=conf, bbox=v.bbox,
                    note=f"{v.cls_name} overran the stop line ({src})",
                ))
        return out
