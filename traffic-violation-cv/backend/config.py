"""Central configuration & shared constants for the traffic-violation pipeline.

Single source of truth — tweak thresholds here, not scattered across modules
(DRY). Paths are resolved relative to the project root so the app runs from
anywhere.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
SAMPLE_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "violations.db"

for _d in (UPLOAD_DIR, ANNOTATED_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Detection model -------------------------------------------------------
# YOLOv8 nano: small + fast, auto-downloads on first use. Swap for a larger
# variant (yolov8s/m/l) for higher accuracy at the cost of speed.
# Resolve to an absolute local path if the weights are already present in the
# project root (avoids a network re-download on every cwd change).
_local_yolo = ROOT / "yolov8n.pt"
YOLO_WEIGHTS = str(_local_yolo) if _local_yolo.exists() else "yolov8n.pt"
# Optional helmet-specialised weights. Drop a custom .pt here to light up
# robust helmet detection; absent -> NoHelmet detector reports "model missing".
HELMET_WEIGHTS = ROOT / "models" / "helmet.pt"
# Optional seatbelt model (YOLOv11, classes no_seatbelt/seat_belt). When present
# NoSeatbeltDetector runs it on the windscreen region of each car.
SEATBELT_WEIGHTS = ROOT / "models" / "seatbelt.pt"
# Optional YOLOv5 license-plate DETECTOR. When present, plates are tight-cropped
# before OCR (big accuracy win). Absent -> OCR runs on the whole vehicle crop.
PLATE_WEIGHTS = ROOT / "models" / "plate.pt"

# --- Deep image restoration (Restormer) ----------------------------------
# Heavy, optional. Layered on top of the fast classical preprocessing and only
# run when requested (meta["deep_restore"]) because CPU inference is slow.
DERAIN_WEIGHTS = ROOT / "models" / "restormer_deraining.pth"
DEBLUR_WEIGHTS = ROOT / "models" / "restormer_motion_deblurring.pth"
RESTORE_MAX_SIDE = 720      # cap longest side before deep restoration (speed)
DEEP_RESTORE_DEFAULT = False  # off by default; toggle per-request

# --- Indian / South-Asian traffic detector (preferred when present) -------
# COCO has no auto-rickshaw class, so three-wheelers get mislabelled as 'car'.
# This YOLOv8 model (arabinda91/yolov8-indian-vehicle, ONNX) adds CNG (auto-
# rickshaw), rickshaw, easybike, leguna, etc. Used as the PRIMARY vehicle
# detector; COCO is kept as a supplement for traffic lights (red-light logic).
INDIAN_WEIGHTS = ROOT / "models" / "indian_vehicle.onnx"
INDIAN_LABELS = ROOT / "models" / "indian_vehicle_labels.txt"
PREFER_INDIAN = True            # use the Indian model first when available
SUPPLEMENT_TRAFFIC_LIGHTS = True  # run COCO too, only to grab traffic lights

DETECT_CONF = 0.35          # min YOLO confidence to keep a detection
IOU_OVERLAP = 0.10          # min IoU to consider two boxes "associated"

# --- COCO classes we care about -------------------------------------------
# Maps raw COCO label -> our normalised road-user category.
VEHICLE_CLASSES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "bicycle",
}

# Maps the Indian model's raw labels -> our normalised categories.
INDIAN_CLASS_MAP = {
    "Car": "car", "MPV": "car", "Van": "car", "ShoppingVan": "car",
    "Bus": "bus", "Leguna": "bus",
    "Truck": "truck", "Pickup": "truck", "PowerTiller": "truck",
    "Motorbike": "motorcycle", "Bicycle": "bicycle",
    "CNG": "auto_rickshaw", "Easybike": "auto_rickshaw",
    "Rickshaw": "rickshaw",
    "Pedestrian": "person",
    # Bhotbhoti (boat) and Wheelbarrow are intentionally ignored.
}

# Normalised road-user categories treated as 'vehicles' across the pipeline.
VEHICLE_CATEGORIES = {
    "car", "motorcycle", "bus", "truck", "bicycle", "auto_rickshaw", "rickshaw",
}
# Motorised vehicles that should carry a number plate (for OCR).
PLATE_BEARING = {"car", "motorcycle", "bus", "truck", "auto_rickshaw"}

PERSON_CLASS = "person"
TRAFFIC_LIGHT_CLASS = "traffic light"
TWO_WHEELERS = {"motorcycle", "bicycle"}

# --- Violation taxonomy (predefined classes) ------------------------------
VIOLATION_TYPES = [
    "no_helmet",
    "triple_riding",
    "no_seatbelt",
    "red_light",
    "stop_line",
    "wrong_side",
    "illegal_parking",
]

# Human-friendly labels + colours (BGR for OpenCV) for annotation.
VIOLATION_META = {
    "no_helmet":       {"label": "No Helmet",        "color": (0, 0, 255)},
    "triple_riding":   {"label": "Triple Riding",    "color": (0, 128, 255)},
    "no_seatbelt":     {"label": "No Seatbelt",      "color": (0, 0, 200)},
    "red_light":       {"label": "Red-Light Jump",   "color": (0, 0, 255)},
    "stop_line":       {"label": "Stop-Line Cross",  "color": (0, 165, 255)},
    "wrong_side":      {"label": "Wrong-Side",       "color": (255, 0, 255)},
    "illegal_parking": {"label": "Illegal Parking",  "color": (255, 0, 0)},
}
