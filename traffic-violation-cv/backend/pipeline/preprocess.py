"""Image preprocessing & enhancement.

Tackles the brief's "robust to varying environmental conditions" requirement:
low light, shadows, rain/haze and motion blur. Everything here is classical CV
(fast, no GPU) and runs before detection. Each step is independent and reports
what it did so the dashboard can show the enhancement provenance.
"""
from __future__ import annotations

import cv2
import numpy as np


def _brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — low value == blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance(image: np.ndarray, deep_derain: bool = False,
            deep_deblur: bool = False) -> tuple[np.ndarray, dict]:
    """Normalise + enhance an input image.

    Fast classical steps always run. The optional Restormer deep stages
    (deep_derain / deep_deblur) run first when enabled - they're slow on CPU
    so they're opt-in per request.

    Returns the processed image and a report dict describing applied steps.
    """
    report: dict = {"steps": []}
    img = image.copy()

    # 0. Deep restoration (optional, heavy) - run on the raw image first.
    if deep_derain or deep_deblur:
        from backend.pipeline import restore
        if deep_deblur:
            img = restore.deblur(img)
            report["steps"].append("restormer_deblur")
        if deep_derain:
            img = restore.derain(img)
            report["steps"].append("restormer_derain")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = _brightness(gray)
    blur = _blur_score(gray)
    report["brightness"] = round(brightness, 1)
    report["sharpness"] = round(blur, 1)

    # 1. Low-light boost via gamma correction when the scene is dim.
    if brightness < 90:
        gamma = 1.6
        lut = np.array([((i / 255.0) ** (1 / gamma)) * 255
                        for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, lut)
        report["steps"].append("low_light_gamma")

    # 2. CLAHE on the L channel — evens out shadows & local contrast.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    report["steps"].append("clahe_shadow_balance")

    # 3. Rain / haze knock-down: gentle denoise preserves edges.
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    report["steps"].append("bilateral_denoise")

    # 4. Motion-blur recovery: unsharp mask only when image is soft.
    if blur < 120:
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        report["steps"].append("unsharp_deblur")

    return img, report
