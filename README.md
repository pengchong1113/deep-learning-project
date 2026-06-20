# Fitness Assistant — Deep Learning Final Project

A deep learning MVP that automatically recognizes fitness exercises from video using a CNN+LSTM architecture.

**IE University | Deep Learning Course**

---

## Overview

Upload a workout video and the system identifies the exercise being performed in real-time. The pipeline combines a pre-trained CNN (MediaPipe BlazePose) for spatial feature extraction with a custom-trained LSTM for temporal action classification.

**Supported exercises:** Squat · Push-up · Sit-up · Pull-up · Jumping Jacks · Jump Rope · Bench Press · Clean & Jerk

---

## Architecture

```
Video frames
    ↓
MediaPipe BlazePose (pre-trained CNN)
— extracts 33 body keypoints (x, y, z) per frame
    ↓
Keypoint sequence  (64 frames × 99 features)
    ↓
Custom LSTM Classifier (2 layers, hidden=128)
— models temporal patterns across frames
    ↓
Exercise prediction + confidence score
```

**Dataset:** Penn Action Dataset (1,163 videos across 8 fitness action classes)
**Val Accuracy:** 80.5%

---

## Project Structure

```
fitness-assistant/
├── data/
│   ├── raw/Penn_Action/        # Penn Action dataset (not in git)
│   └── processed/keypoints/    # Extracted keypoints (not in git)
├── model/
│   ├── lstm_classifier.py      # Model definition
│   └── saved_model/            # Trained weights (not in git)
├── scripts/
│   ├── extract_keypoints.py    # Step 1: extract MediaPipe keypoints
│   ├── train.py                # Step 2: train LSTM classifier
│   └── make_test_video.py      # Utility: convert frames to .mp4 for testing
├── backend/
│   ├── main.py                 # FastAPI app
│   └── predictor.py            # Inference pipeline
├── frontend/
│   └── app.py                  # Streamlit UI
├── models/                     # MediaPipe model file (not in git)
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate           # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Prepare data

Download [Penn Action Dataset](http://dreamdragon.github.io/PennAction/) and place it under:
```
data/raw/Penn_Action/frames/
data/raw/Penn_Action/labels/
```

Extract keypoints (downloads MediaPipe model automatically, ~4MB):
```bash
python scripts/extract_keypoints.py
```

### Step 2 — Train the model

```bash
python scripts/train.py
# Optional flags:
# --epochs 60  --batch_size 32  --lr 1e-3  --hidden 128  --layers 2
```

Trained model is saved to `model/saved_model/best_model.pt`.

### Step 3 — Run the app

Open **two terminals**, both with the virtual environment activated:

```bash
# Terminal 1 — Backend
uvicorn backend.main:app --reload

# Terminal 2 — Frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser, upload a workout video, and click **Analyze**.

---

## Model Performance

| Exercise | Accuracy |
|----------|----------|
| Jumping Jacks | 98.2% |
| Pull-up | 88.1% |
| Push-up | 85.0% |
| Clean & Jerk | 86.7% |
| Jump Rope | 78.6% |
| Squat | 74.4% |
| Bench Press | 66.2% |
| Sit-up | 66.0% |
| **Overall** | **80.5%** |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Pose estimation | MediaPipe BlazePose |
| Deep learning | PyTorch |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Data processing | OpenCV, NumPy, SciPy |
