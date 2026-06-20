"""Evidence generation — annotate images with detections & violations.

Draws detection boxes (thin, grey) and violation boxes (thick, colour-coded
per type) with labels + confidence, plus any recognised plate. Output is the
court-of-public-opinion-ready evidence image.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend import config
from backend.pipeline.types import Frame


def _put_label(img, text, x, y, color, scale=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    y_box = max(0, y - th - 6)
    cv2.rectangle(img, (x, y_box), (x + tw + 6, y_box + th + 6), color, -1)
    cv2.putText(img, text, (x + 3, y_box + th + 1), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def annotate(frame: Frame) -> np.ndarray:
    """Return a new annotated BGR image (does not mutate the original)."""
    img = frame.original.copy()

    # Faint detection boxes for context.
    for idx, det in enumerate(frame.detections):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
        tag = det.cls_name
        if idx in frame.plates:
            tag += f" [{frame.plates[idx]}]"
        _put_label(img, tag, x1, y1, (110, 110, 110), 0.4)

    # Bold violation boxes on top.
    for v in frame.violations:
        meta = config.VIOLATION_META.get(v.vtype, {"label": v.vtype,
                                                    "color": (0, 0, 255)})
        color = meta["color"]
        x1, y1, x2, y2 = v.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{meta['label']} {v.confidence:.0%}"
        if v.plate:
            label += f" | {v.plate}"
        _put_label(img, label, x1, max(y1, 18), color, 0.55)

    # Footer banner.
    banner = f"Violations: {len(frame.violations)}  |  Objects: {len(frame.detections)}"
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 26), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, banner, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img
