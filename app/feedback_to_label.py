import os
import csv

# === CẤU HÌNH ===
LOG_FILE = "app/tracking_log.csv"
IMAGE_DIR = "app/images"
LABEL_DIR = "app/labels"

# Tạo thư mục nếu chưa có
os.makedirs(LABEL_DIR, exist_ok=True)

# Đọc tracking_log.csv
with open(LOG_FILE, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 8:
            continue  # Bỏ qua dòng không đầy đủ
        frame_id, track_id, cls, x1, y1, x2, y2, feedback = row

        if feedback.strip().lower() != "wrong":
            continue  # Chỉ lấy dòng bị gán nhãn sai

        # Tính bbox YOLO format (cx, cy, w, h) – ảnh gốc là ảnh đã crop nên w,h là full
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w = x2 - x1
        h = y2 - y1
        cx = w / 2
        cy = h / 2

        # Normalize về (0-1) theo kích thước ảnh crop
        norm_cx = cx / w
        norm_cy = cy / h
        norm_w = 1.0
        norm_h = 1.0

        # Ghi file label tương ứng
        label_path = os.path.join(LABEL_DIR, f"pizza_{track_id}.txt")
        with open(label_path, "w") as label_file:
            class_id = 0  # Pizza
            label_file.write(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        print(f"✅ Tạo label cho pizza_{track_id}.jpg")
