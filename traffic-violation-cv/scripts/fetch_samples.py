"""Generate synthetic traffic sample images for offline demos.

Real YOLO inference needs real photos, but for a guaranteed-runnable demo (and
for exercising the mock detector) these synthetic scenes are enough to walk the
full pipeline. For real testing, drop actual traffic JPEGs into data/samples/.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend import config


def _scene(idx: int) -> np.ndarray:
    h, w = 540, 960
    img = np.full((h, w, 3), (130, 130, 130), np.uint8)          # road grey
    cv2.rectangle(img, (0, 0), (w, 180), (170, 140, 90), -1)      # sky-ish band
    # lane markings
    for x in range(40, w, 120):
        cv2.rectangle(img, (x, h // 2 - 4), (x + 60, h // 2 + 4), (255, 255, 255), -1)
    # stop line
    cv2.rectangle(img, (0, int(h * 0.55)), (w, int(h * 0.55) + 6), (255, 255, 255), -1)
    # a couple of blocky "vehicles"
    cv2.rectangle(img, (360, 360), (520, 470), (40, 40, 200), -1)
    cv2.rectangle(img, (720, 330), (880, 430), (60, 160, 60), -1)
    # a "traffic light" (red on)
    cv2.rectangle(img, (880, 40), (910, 130), (20, 20, 20), -1)
    cv2.circle(img, (895, 60), 10, (0, 0, 255), -1)
    cv2.putText(img, f"synthetic-scene-{idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main(n: int = 4) -> None:
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        path = config.SAMPLE_DIR / f"sample_{i}.jpg"
        cv2.imwrite(str(path), _scene(i))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
