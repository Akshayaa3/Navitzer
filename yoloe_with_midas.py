import cv2
import torch
import numpy as np
import random
import pyttsx3
from ultralytics import YOLOE

# ─────────────── Configuration ───────────────
MODEL_PATH = "yoloe-11s-seg.pt"
CLASS_NAMES = [
    "person", "cellphone", "chair", "laptop", "bottle",
    "windows", "door", "table", "tv", "sofa", "bed",
    "fan", "light", "keyboard", "mouse", "stairs",
    "printer", "cabinet", "shelf", "couch", "plant"
]
CONFIDENCE_THRESHOLD = 0.5
MIN_BOX_AREA = 1000
VOICE_ALERT_THRESHOLD_M = 2.0  # meters

# ─────────────── Initialize Models ───────────────
# YOLOE
yoloe = YOLOE(MODEL_PATH)
yoloe.set_classes(CLASS_NAMES, yoloe.get_text_pe(CLASS_NAMES))

# MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Assign random colors to each class
class_colors = {
    name: (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
    for name in CLASS_NAMES
}

# Text-to-Speech engine
engine = pyttsx3.init()
engine.say("Voice engine initialized.")
engine.runAndWait()
last_alert = None

# ─────────────── Helper: Depth Estimation ───────────────
def estimate_depth(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(cv2.resize(rgb, (384, 384))).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
    return (norm_depth * 255).astype(np.uint8), depth

# Calibration: measured MiDaS value at 1 meter (manual step)
REFERENCE_DEPTH = 0.7
SCALE_FACTOR = 1.0 / REFERENCE_DEPTH

# ─────────────── Main Loop ───────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

        depth_vis, raw_depth = estimate_depth(frame)
        results = yoloe.predict(frame)
        boxes = results[0].boxes

        detected_objects = []
        closest_obj = None
        closest_depth = float('inf')

        for box in boxes:
            conf = float(box.conf.item())
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            try:
                class_id = int(box.cls.item())
                class_name = CLASS_NAMES[class_id]
            except (IndexError, ValueError):
                class_name = "unknown"

            color = class_colors.get(class_name, (0, 255, 0))
            depth_crop = raw_depth[y1:y2, x1:x2]
            mean_depth = np.mean(depth_crop) if depth_crop.size else 0
            distance_m = mean_depth * SCALE_FACTOR
            label = f"{class_name} ({distance_m:.2f}m)"

            print(f"Detected: {class_name}, Distance: {distance_m:.2f}m")
            detected_objects.append((class_name, distance_m))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if distance_m < closest_depth:
                closest_depth = distance_m
                closest_obj = {
                    "label": class_name,
                    "bbox": (x1, y1, x2, y2),
                    "distance": distance_m
                }

        # Display summary
        cv2.rectangle(frame, (5, 5), (310, 25 + 20 * len(detected_objects)), (0, 0, 0), -1)
        cv2.putText(frame, f"Objects Detected: {len(detected_objects)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        for idx, (name, dist) in enumerate(detected_objects):
            cv2.putText(frame, f"{name}: {dist:.2f}m", (10, 40 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Voice alert for nearest
        if closest_obj and closest_depth < VOICE_ALERT_THRESHOLD_M:
            x1, y1, x2, y2 = closest_obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"Nearest: {closest_obj['label']} ({closest_obj['distance']:.2f}m)",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if last_alert != closest_obj['label']:
                speech = f"Caution. {closest_obj['label']} is {closest_obj['distance']:.2f} meters away."
                print("Voice Alert:", speech)
                engine.say(speech)
                engine.runAndWait()
                last_alert = closest_obj['label']

        # Combine and show
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        combined = np.hstack((frame, depth_colored))

        cv2.imshow("YOLOE + MiDaS Navigation", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
