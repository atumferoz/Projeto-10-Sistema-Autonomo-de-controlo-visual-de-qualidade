import json
from pathlib import Path

LABELS = ["Parafuso", "Porca"]

def parse_label_file(label_path):
    detections = []
    with open(label_path) as f:
        for line in f:
            cls, cx, cy, w, h, angle, conf = map(float, line.split())
            detections.append({
                "class_id": int(cls),
                "class_name": LABELS[int(cls)],
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "angle": angle,
                "confidence": conf
            })
    return detections

all_results = {}

label_dir = Path("inference/run1/labels")
for label_file in label_dir.glob("*.txt"):
    all_results[label_file.stem] = parse_label_file(label_file)

with open("detections.json", "w") as f:
    json.dump(all_results, f, indent=2)
