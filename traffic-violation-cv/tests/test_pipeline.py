"""Smoke + unit tests for the pipeline. Run: python -m pytest -q

These run in 'demo mode' (mock detector) so they pass without the heavy ML
stack installed — perfect for CI and quick sanity checks.
"""
from __future__ import annotations

import numpy as np

from backend.pipeline import engine
from backend.pipeline.types import (
    Detection, Frame, iou, overlap_ratio, union_bbox,
)
from backend.pipeline.violations.riders import TripleRidingDetector


def _img():
    return np.full((540, 960, 3), 120, np.uint8)


def test_geometry_helpers():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert overlap_ratio((0, 0, 10, 10), (0, 0, 20, 20)) == 1.0
    assert union_bbox([(0, 0, 5, 5), (10, 10, 20, 20)]) == (0, 0, 20, 20)


def test_pipeline_runs_and_returns_metrics():
    frame, metrics = engine.process(_img())
    assert "timings_ms" in metrics
    assert metrics["n_objects"] >= 0
    assert isinstance(frame.violations, list)
    # preprocess report attached
    assert "preprocess" in frame.meta


def test_triple_riding_detector_fires():
    # build a frame with a motorcycle + 3 overlapping persons
    bike = Detection("motorcycle", 0.9, (100, 300, 260, 480))
    riders = [Detection("person", 0.8, (110 + i * 20, 280, 170 + i * 20, 460))
              for i in range(3)]
    frame = Frame(image=_img(), original=_img(),
                  detections=[bike, *riders])
    vios = TripleRidingDetector().run(frame)
    assert any(v.vtype == "triple_riding" for v in vios)


def test_no_violation_when_two_riders():
    bike = Detection("motorcycle", 0.9, (100, 300, 260, 480))
    riders = [Detection("person", 0.8, (110 + i * 20, 280, 170 + i * 20, 460))
              for i in range(2)]
    frame = Frame(image=_img(), original=_img(), detections=[bike, *riders])
    vios = TripleRidingDetector().run(frame)
    assert not any(v.vtype == "triple_riding" for v in vios)
