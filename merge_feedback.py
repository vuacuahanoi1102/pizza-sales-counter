import os
import shutil

# === CẤU HÌNH ===
IMAGE_SRC = "app/images"                      # Ảnh người dùng feedback
LABEL_SRC = "app/labels"                      # Label tương ứng do người dùng tạo
IMAGE_DST = "dataset/images/train"            # Dataset YOLO - ảnh train
LABEL_DST = "dataset/labels/train"            # Dataset YOLO - label train

# Tạo thư mục đích nếu chưa có
os.makedirs(IMAGE_DST, exist_ok=True)
os.makedirs(LABEL_DST, exist_ok=True)

# Lặp qua toàn bộ ảnh trong app/images/
for fname in os.listdir(IMAGE_SRC):
    if not fname.endswith(".jpg"):
        continue
    img_src_path = os.path.join(IMAGE_SRC, fname)
    label_src_path = os.path.join(LABEL_SRC, fname.replace(".jpg", ".txt"))

    img_dst_path = os.path.join(IMAGE_DST, fname)
    label_dst_path = os.path.join(LABEL_DST, fname.replace(".jpg", ".txt"))

    if not os.path.exists(label_src_path):
        print(f"⚠️ Không tìm thấy label cho {fname}, bỏ qua.")
        continue

    shutil.copy(img_src_path, img_dst_path)
    shutil.copy(label_src_path, label_dst_path)
    print(f"✅ Copied {fname} + label")

print("🎯 DONE. Đã merge toàn bộ feedback vào dataset YOLO.")
