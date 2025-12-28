from ultralytics import YOLO
from pathlib import Path
import json
import csv
import cv2
import numpy as np
from collections import defaultdict

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "runs/obb/parafuso_porca_train_v1/weights/best.pt"
IMAGE_DIR = "data/parafuso_porcas/images/val"
OUT_DIR = Path("outputs")
CONF_THRES = 0.5

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "annotated").mkdir(exist_ok=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# RUN INFERENCE
# ----------------------------
results = model(
    source=IMAGE_DIR,
    conf=CONF_THRES,
    imgsz=640,
    device=0
)

# ----------------------------
# COLLECT RAW DETECTIONS
# ----------------------------
detections = []

for r in results:
    image_name = Path(r.path).name

    if r.obb is None:
        continue

    obb = r.obb.cpu()

    for i in range(len(obb.cls)):
        cls_id = int(obb.cls[i])
        conf = float(obb.conf[i])

        cx, cy, w, h, angle = obb.xywhr[i].tolist()

        detections.append({
            "image": image_name,
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": conf,
            "cx_px": cx,
            "cy_px": cy,
            "w_px": w,
            "h_px": h,
            "angle_rad": angle,
            "long_side_px": max(w, h),
            "short_side_px": min(w, h)
        })

# ----------------------------
# SAVE RAW JSON
# ----------------------------
with open(OUT_DIR / "detections_raw.json", "w") as f:
    json.dump(detections, f, indent=2)

print("✔ Saved detections_raw.json")

# ----------------------------
# GROUP BY IMAGE & COMPARE BOLTS
# ----------------------------
by_image = defaultdict(list)

for d in detections:
    if d["class_name"] == "Parafuso":
        by_image[d["image"]].append(d)

for image_name, bolts in by_image.items():
    if len(bolts) < 2:
        for b in bolts:
            b["length_label"] = "ONLY_ONE"
        continue

    max_len = max(b["long_side_px"] for b in bolts)

    for b in bolts:
        if abs(b["long_side_px"] - max_len) < 1e-3:
            b["length_label"] = "LONGEST"
        else:
            b["length_label"] = "SHORTER"

# Add default label for Porcas
for d in detections:
    if d["class_name"] != "Parafuso":
        d["length_label"] = "N/A"

# ----------------------------
# SAVE CSV
# ----------------------------
csv_path = OUT_DIR / "results.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=detections[0].keys())
    writer.writeheader()
    writer.writerows(detections)

print("✔ Saved results.csv")

# ----------------------------
# DRAW ANNOTATED IMAGES
# ----------------------------
for r in results:
    img = cv2.imread(r.path)
    image_name = Path(r.path).name

    if r.obb is None:
        continue

    obb = r.obb.cpu()
    boxes = obb.xyxyxyxy.numpy()  # shape: (N, 8)

    for i in range(len(obb.cls)):
        cls_id = int(obb.cls[i])
        class_name = model.names[cls_id]
        conf = float(obb.conf[i])

        # match detection entry
        det = next(
            d for d in detections
            if d["image"] == image_name
            and abs(d["confidence"] - conf) < 1e-3
            and d["class_name"] == class_name
        )

        pts = boxes[i].reshape(-1, 2).astype(np.int32)

        # Color logic
        if det["length_label"] == "LONGEST":
            color = (0, 0, 255)      # Red
        elif det["length_label"] == "SHORTER":
            color = (0, 255, 255)    # Yellow
        else:
            color = (0, 255, 0)      # Green

        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        label = f"{class_name} | {det['length_label']} | {conf:.2f}"
        text_pos = (int(det["cx_px"]), int(det["cy_px"]))

        cv2.putText(
            img,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imwrite(str(OUT_DIR / "annotated" / image_name), img)

print("✔ Saved annotated images")
