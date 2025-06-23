# 🍕 Pizza Sales Counting System

This project is an intelligent monitoring system to count pizzas sold in a restaurant environment, based on video input.  
It uses **YOLOv8 for object detection**, **ByteTrack for multi-object tracking**, and counts a pizza as "sold" when it is placed into a pizza box and disappears from the frame.

> ✅ *This is a take-home assessment for EATLAB.*  
> 🙏 *Special thanks to OpenAI's GPT-4 for guidance throughout the entire development process. As someone new to computer vision (CV), I relied heavily on GPT to learn and build this system from scratch. Prior to this project, my experience was mostly limited to PyTorch classification on static images.*

---

## 📁 Project Structure

```
pizza_project/
├── app/
│   ├── feedback_to_label.py       # Convert user feedback into YOLO label
│   ├── tracking_log.csv           # Tracking results and user feedback
│   └── images/                    # Cropped images of incorrect predictions
│
├── dataset/
│   ├── images/train/              # YOLO training images
│   ├── labels/train/              # YOLO labels
│   └── dataset.yaml               # YOLO config
│
├── runs/                          # YOLO training outputs
├── video1.mp4                     # Input video to analyze
├── pizza_tracker.py              # Main script to detect, track, and count sales
├── merge_feedback.py             # Merge corrected feedback into dataset
├── requirements.txt              # Python dependencies
└── README.md                     # You're reading it!
```

---

## 🚀 Features

- 🔍 Detects `pizza` and `box` using a YOLOv8 model trained on custom data.
- 🧠 Tracks individual pizzas using ByteTrack (via YOLOX).
- 🧾 Counts pizzas as sold when:
  - They're placed inside a box (based on IOU overlap),
  - And disappear from the scene.
- 🛠 Supports user feedback:
  - Users can correct false detections by labeling cropped images,
  - System can be refined via `merge_feedback.py` + retraining.

---

## ⚙️ Setup Instructions (No Docker)

### 🐍 1. Create virtual environment (recommended)
```bash
python3.10 -m venv pizza-env
source pizza-env/bin/activate  # or pizza-env\Scripts\activate on Windows
```

### 📦 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 📂 3. Clone & install ByteTrack manually
```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip install -r requirements.txt
pip install -v -e .
cd ..
```

> 💡 This ensures `yolox.tracker.byte_tracker` is available.

---

## ▶️ Run the System

```bash
python pizza_tracker.py
```

> 🎥 This will analyze `video1.mp4`, show detections + real-time sale count.

---

## 🧪 Feedback Loop for Model Improvement

Incorrect Prediction? Here's how to fix it:

1. Open `app/images/` to find the wrong detection.
2. Label the image using [LabelImg](https://github.com/tzutalin/labelImg) (YOLO format).
3. Save the label file (`.txt`) to `app/labels/`.
4. Merge feedback into training set:
```bash
python merge_feedback.py
```
5. Retrain YOLO model (optional):
```bash
yolo detect train data=dataset/dataset.yaml model=yolov8n.pt epochs=20
```

---

## 📦 Requirements

- Python 3.10
- [YOLOv8](https://docs.ultralytics.com/) (`ultralytics`)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (manual install)
- PyTorch, OpenCV, NumPy, etc.

> ✅ All dependencies are listed in `requirements.txt`

---

## 👋 Acknowledgement

Built with love, hard work, and a lot of help from ChatGPT 🙏  
This project helped me understand the full pipeline of object detection, tracking, and continuous feedback-based improvement — a truly eye-opening experience!
