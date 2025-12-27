import tensorflow as tf
import numpy as np
import cv2
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "/workspace/model"
IMAGE_PATH = "/workspace/images/test.jpeg"
CLASSES_PATH = "/workspace/classes.txt"
OUTPUT_PATH = "/workspace/output/result.jpg"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.6
# ---------------------------------------

# Load class names
with open(CLASSES_PATH) as f:
    CLASS_NAMES = [c.strip() for c in f.readlines()]

# Load model
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

# Load original image
orig = cv2.imread(IMAGE_PATH)
h, w, _ = orig.shape

# Preprocess image
img = tf.io.read_file(IMAGE_PATH)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = tf.expand_dims(img, 0)

# Run inference
outputs = infer(x=img)
pred = outputs["output_0"].numpy()[0]  # (25200, 7)

boxes = []
scores = []
classes = []

# Parse predictions
for det in pred:
    cx, cy, bw, bh, obj_conf, cls_id, cls_conf = det
    score = obj_conf * cls_conf

    if score < CONF_THRESHOLD:
        continue

    # Convert to corner format (normalized)
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes.append([y1, x1, y2, x2])  # TF expects y1,x1,y2,x2
    scores.append(score)
    classes.append(int(cls_id))

# If nothing detected
if len(boxes) == 0:
    print("⚠ No detections")
    cv2.imwrite(OUTPUT_PATH, orig)
    exit()

# Convert to tensors
boxes = tf.constant(boxes, dtype=tf.float32)
scores = tf.constant(scores, dtype=tf.float32)

# Apply NMS
selected = tf.image.non_max_suppression(
    boxes,
    scores,
    max_output_size=50,
    iou_threshold=IOU_THRESHOLD,
    score_threshold=CONF_THRESHOLD
)

# Draw final detections
for i in selected.numpy():
    y1, x1, y2, x2 = boxes[i].numpy()
    cls_id = classes[i]
    score = scores[i].numpy()

    # Convert to pixel coords
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        orig,
        f"{label} {score:.2f}",
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

# Save output
os.makedirs("/workspace/output", exist_ok=True)
cv2.imwrite(OUTPUT_PATH, orig)
print("✅ Saved result to:", OUTPUT_PATH)
