# Presentation

## Title
**AutoTraffic Vision — AI-Powered Traffic Violation Detection from Photographic Evidence**

## Tagline
Computer Vision for Smart Traffic Enforcement

## Description
AutoTraffic Vision is a scalable computer-vision system that automatically
processes traffic surveillance images, detects vehicles and road users,
identifies and classifies seven types of traffic violations, recognizes license
plates via OCR, generates annotated evidence with metadata and timestamps, and
provides analytics, searchable records, and performance evaluation — drastically
reducing manual inspection effort for traffic enforcement.

## Files
| File | Format | Use |
|---|---|---|
| `AutoTraffic_Vision.pptx` | PowerPoint | Editable native slide deck (14 slides) |
| `slides.html` | HTML | Browser-based slide deck (keyboard-navigable) |
| `build_pptx.py` | Python | Regenerates the `.pptx` (`python docs/build_pptx.py`) |

To rebuild the PowerPoint: `pip install python-pptx` then
`python docs/build_pptx.py`.

## Slide outline (14 slides)
1. **Title** — project name, tagline, one-line description
2. **The Problem** — manual inspection is slow, inconsistent, doesn't scale
3. **Solution Overview** — end-to-end CV pipeline, robust to weather/quality
4. **System Architecture** — pipeline stages + tech stack
5. **Image Preprocessing** — low-light/rain/shadow/blur handling + Restormer
6. **Vehicle & Road-User Detection** — COCO + Indian-vehicle YOLO fusion
7. **The 7 Violation Types** — helmet, triple-riding, seatbelt, red-light,
   stop-line, wrong-side, illegal-parking (with confidence scores)
8. **License Plate Recognition** — two-stage ANPR (YOLOv5 + EasyOCR)
9. **Evidence Generation** — annotated images + SQLite metadata + timestamps
10. **Analytics & Reporting** — stats, trends, top plates, searchable records
11. **Performance Evaluation** — Accuracy / Precision / Recall / F1 / mAP@0.5 +
    computational efficiency
12. **Engineering Quality** — real models, false-positive guards, honest abstention
13. **Limitations & Future Work** — honest current limits + roadmap
14. **Summary & Impact** — closing
