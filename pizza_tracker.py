import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
import os
import csv

# === HÀM TÍNH IOU === tính diện tích vùng giao nhau
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area

    return inter_area / union

# === CẤU HÌNH ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "video1.mp4"
class_names = ["pizza", "box"]

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Không mở được video. Kiểm tra lại đường dẫn.")
    exit()

args = Namespace(
    track_thresh=0.3,
    match_thresh=0.8,
    buffer_size=30,
    track_buffer=30,
    min_box_area=10,
    mot20=False
)
tracker = BYTETracker(args)

frame_id = 0
pizza_tracks = {}
sold_pizzas = set()
sold_count = 0

# === TẠO THƯ MỤC VÀ FILE LOG ===
os.makedirs("app/images", exist_ok=True)
log_file = open("app/tracking_log.csv", mode='a', newline='')
log_writer = csv.writer(log_file)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # === DETECT ===
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append([x1, y1, x2, y2, conf, cls])

    if len(detections) == 0:
        dets = torch.empty((0, 6), dtype=torch.float32)
        cls_list = torch.empty((0,))
    else:
        dets = torch.from_numpy(np.array(detections, dtype=np.float32)).float()
        cls_list = dets[:, 5]
        dets_for_track = dets[:, :5]

    img_h, img_w = frame.shape[:2]
    online_targets = tracker.update(dets[:, :5] if len(dets) else dets, (img_h, img_w), frame_id)

    # === GÁN LẠI CLASS CHO TRACK ===
    for t, cls in zip(online_targets, cls_list):
        t.cls = int(cls.item())

    # === TÁCH PIZZA VÀ BOX ===
    pizzas = [t for t in online_targets if t.cls == 0]
    boxes = [t for t in online_targets if t.cls == 1]

    # === GHI NHỚ PIZZA TRACKING ===
    for p in pizzas:
        pid = p.track_id
        in_box = any(iou(p.tlbr, b.tlbr) > 0.5 for b in boxes)
        pizza_tracks[pid] = {
            "last_seen": frame_id,
            "in_box": in_box,
            "tlbr": p.tlbr
        }

    # === PHÁT HIỆN PIZZA BIẾN MẤT SAU KHI VÀO BOX ===
    to_remove = []
    for pid, info in pizza_tracks.items():
        if frame_id - info["last_seen"] > 15 and info["in_box"] and pid not in sold_pizzas:
            sold_count += 1
            sold_pizzas.add(pid)
            print(f"🍕 Pizza SOLD! ID={pid}")

            # === GHI LOG ===
            x1, y1, x2, y2 = map(int, info["tlbr"])
            log_writer.writerow([info["last_seen"], pid, "pizza", x1, y1, x2, y2, 1.0])

            # === CROP ẢNH ===
            crop = frame[y1:y2, x1:x2]
            cv2.imwrite(f"app/images/pizza_{pid}.jpg", crop)

            to_remove.append(pid)
    for pid in to_remove:
        del pizza_tracks[pid]

    # === VẼ KẾT QUẢ ===
    for t in online_targets:
        x1, y1, x2, y2 = map(int, t.tlbr)
        track_id = t.track_id
        cls = t.cls
        label = f"{class_names[cls]}-{track_id}"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === VẼ SỐ PIZZA ĐÃ BÁN ===
    cv2.putText(frame, f"Sold: {sold_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("🍕 Pizza Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
