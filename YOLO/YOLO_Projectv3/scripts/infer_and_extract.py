from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json
import csv
import cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "runs_V2/obb/parafuso_porca_train_v1/weights/best.pt"
IMAGE_SOURCE = "data/parafuso_porcas/images/val"
CONF_THRES = 0.5
IMG_SIZE = 640
DEVICE = 0

# Calibration (CHANGE THIS after real measurement)
PX_PER_MM = 10.0  # example: 10 pixels = 1 mm

CLASS_NAMES = {
    0: "Parafuso",
    1: "Porca"
}

# =========================
# OUTPUT DIRECTORIES
# =========================
RUN_NAME = datetime.now().strftime("inference_%Y-%m-%d_%H-%M-%S")
BASE_OUTPUT_DIR = Path("runs") / RUN_NAME
ANNOTATED_DIR = BASE_OUTPUT_DIR / "annotated"

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# RUN INFERENCE
# =========================
results = model(
    source=IMAGE_SOURCE,
    conf=CONF_THRES,
    imgsz=IMG_SIZE,
    device=DEVICE
)

all_detections = []

# =========================
# PROCESS RESULTS
# =========================
for r in results:
    image_path = Path(r.path)
    image_name = image_path.name

    img = cv2.imread(str(image_path))
    if img is None or r.obb is None:
        continue

    boxes = r.obb.cpu()
    detections = []

    # ---- Extract detections
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        cx, cy, w, h, angle = boxes.xywhr[i].tolist()

        # Convert to mm
        w_mm = w / PX_PER_MM
        h_mm = h / PX_PER_MM
        short_mm = min(w_mm, h_mm)
        long_mm = max(w_mm, h_mm)

        detections.append({
            "image": image_name,
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": conf,
            "cx_px": cx,
            "cy_px": cy,
            "w_mm": round(w_mm, 2),
            "h_mm": round(h_mm, 2),
            "short_mm": round(short_mm, 2),
            "long_mm": round(long_mm, 2),
            "angle_deg": round(angle, 2),
            "box_index": i
        })

    # ---- Analyze parafusos only
    parafusos = [d for d in detections if d["class_name"] == "Parafuso"]

    if len(parafusos) == 1:
        parafusos[0]["size_label"] = "ONLY"
    elif len(parafusos) > 1:
        sorted_p = sorted(parafusos, key=lambda x: x["long_mm"])
        sorted_p[0]["size_label"] = "SHORTEST"
        sorted_p[-1]["size_label"] = "LONGEST"
        for mid in sorted_p[1:-1]:
            mid["size_label"] = "IN-BETWEEN"

    # ---- Draw annotations
    for d in detections:
        i = d["box_index"]
        pts = boxes.xyxyxyxy[i].numpy().reshape(-1, 2).astype(int)

        # Color logic
        color = (0, 255, 255)  # yellow default

        if d["class_name"] == "Parafuso":
            if d.get("size_label") == "SHORTEST":
                color = (255, 0, 0)      # blue
            elif d.get("size_label") == "LONGEST":
                color = (0, 255, 0)      # green
            elif d.get("size_label") == "ONLY":
                color = (255, 0, 255)    # purple
            elif d.get("size_label") == "IN-BETWEEN":
                color = (0, 165, 255)    # orange

        cv2.polylines(img, [pts], True, color, 2)

        label = (
            f"{d['class_name']} "
            f"{d.get('size_label','')} "
            f"{d['long_mm']}mm "
            f"{d['confidence']:.2f}"
        )

        cv2.putText(
            img,
            label.strip(),
            (int(d["cx_px"]), int(d["cy_px"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        all_detections.append(d)

    # ---- Save annotated image
    cv2.imwrite(str(ANNOTATED_DIR / image_name), img)

# =========================
# SAVE JSON
# =========================
json_path = BASE_OUTPUT_DIR / "detections.json"
with open(json_path, "w") as f:
    json.dump(all_detections, f, indent=2)

# =========================
# SAVE CSV
# =========================
csv_path = BASE_OUTPUT_DIR / "detections.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
    writer.writeheader()
    writer.writerows(all_detections)

print(f"\n✅ Inference complete")
print(f"📁 Results saved to: {BASE_OUTPUT_DIR}")
print(f"🖼 Annotated images: {ANNOTATED_DIR}")
print(f"📄 CSV: {csv_path}")
print(f"📄 JSON: {json_path}")
