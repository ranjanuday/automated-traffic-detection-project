"""Deep image restoration — Restormer deraining & motion-deblurring.

Optional heavy stage layered on top of the fast classical preprocessing. Loads
the official Restormer weights (config.DERAIN_WEIGHTS / DEBLUR_WEIGHTS) lazily
and runs them on CPU. Because full-res transformer inference is slow on CPU, we
restore at a capped resolution (config.RESTORE_MAX_SIDE) then map back.

Everything degrades gracefully: missing weights / torch -> returns the input
unchanged so the pipeline always completes.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend import config

_models: dict[str, object] = {}
_failed: set[str] = set()


def status() -> str:
    """Report which deep restorers are available."""
    avail = []
    if config.DERAIN_WEIGHTS.exists():
        avail.append("derain")
    if config.DEBLUR_WEIGHTS.exists():
        avail.append("deblur")
    return "restormer:" + ",".join(avail) if avail else "unavailable"


def _load(kind: str):
    """Lazily build a Restormer and load the task weights (cached)."""
    if kind in _models:
        return _models[kind]
    if kind in _failed:
        return None
    weights = (config.DERAIN_WEIGHTS if kind == "derain"
               else config.DEBLUR_WEIGHTS)
    if not weights.exists():
        _failed.add(kind)
        return None
    try:
        import torch
        from backend.pipeline.restormer_arch import Restormer
        net = Restormer()  # default config matches both task checkpoints
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state, strict=True)
        net.eval()
        _models[kind] = net
        return net
    except Exception as exc:  # noqa: BLE001
        print(f"[restore] {kind} model load failed: {exc}")
        _failed.add(kind)
        return None


def _run(net, image: np.ndarray) -> np.ndarray:
    """Run a Restormer on one BGR image, restoring at capped resolution."""
    import torch

    h0, w0 = image.shape[:2]
    scale = min(1.0, config.RESTORE_MAX_SIDE / max(h0, w0))
    work = cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else image

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ten = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    # Restormer needs H,W divisible by 8 -> reflect-pad then crop back.
    _, _, h, w = ten.shape
    factor = 8
    ph, pw = (factor - h % factor) % factor, (factor - w % factor) % factor
    ten = torch.nn.functional.pad(ten, (0, pw, 0, ph), mode="reflect")

    with torch.no_grad():
        out = net(ten)
    out = torch.clamp(out, 0, 1)[:, :, :h, :w]
    res = (out.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        res = cv2.resize(res, (w0, h0), interpolation=cv2.INTER_CUBIC)
    return res


def derain(image: np.ndarray) -> np.ndarray:
    net = _load("derain")
    return _run(net, image) if net is not None else image


def deblur(image: np.ndarray) -> np.ndarray:
    net = _load("deblur")
    return _run(net, image) if net is not None else image
