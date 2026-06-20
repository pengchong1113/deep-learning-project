"""
Inference pipeline: video file → keypoints → LSTM → prediction.
Loaded once at startup and reused across requests.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from model.lstm_classifier import FitnessLSTM, ACTION_LABELS, SEQ_LEN
from backend.rep_counter import count_reps

MODEL_PATH    = Path('model/saved_model/best_model.pt')
LANDMARKER_PATH = Path('models/pose_landmarker_lite.task')


class Predictor:
    def __init__(self):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model      = self._load_model()
        self.landmarker = self._load_landmarker()

    def _load_model(self) -> FitnessLSTM:
        ckpt  = torch.load(MODEL_PATH, map_location=self.device)
        args  = ckpt.get('args', {})
        model = FitnessLSTM(
            hidden_size=args.get('hidden', 128),
            num_layers=args.get('layers', 2),
            dropout=0.0,   # no dropout at inference
        ).to(self.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model

    def _load_landmarker(self):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _extract_keypoints(self, video_path: str):
        """Returns (frames, 99) flattened array and (frames, 33, 3) full array."""
        cap = cv2.VideoCapture(video_path)
        flat_seq = []
        full_seq = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result  = self.landmarker.detect(mp_img)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                kp  = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            else:
                kp = np.zeros((33, 3), dtype=np.float32)

            full_seq.append(kp)
            flat_seq.append(kp.flatten())

        cap.release()
        return (
            np.array(flat_seq, dtype=np.float32),   # (frames, 99)
            np.array(full_seq, dtype=np.float32),   # (frames, 33, 3)
        )

    def _pad_or_truncate(self, kp: np.ndarray) -> np.ndarray:
        n = len(kp)
        if n >= SEQ_LEN:
            return kp[:SEQ_LEN]
        pad = np.zeros((SEQ_LEN - n, kp.shape[1]), dtype=np.float32)
        return np.concatenate([kp, pad], axis=0)

    def predict(self, video_path: str) -> dict:
        kp_flat, kp_full = self._extract_keypoints(video_path)

        if len(kp_flat) == 0:
            return {'error': 'No pose detected in video'}

        # Action classification
        kp_padded = self._pad_or_truncate(kp_flat)
        x         = torch.tensor(kp_padded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

        top_idx = int(np.argmax(probs))
        action  = ACTION_LABELS[top_idx]

        # Rep counting on the full (unpadded) keypoint sequence
        rep_result = count_reps(kp_full, action)

        return {
            'action':         action,
            'confidence':     round(probs[top_idx], 4),
            'reps':           rep_result['reps'],
            'probabilities':  {
                label: round(prob, 4)
                for label, prob in zip(ACTION_LABELS, probs)
            },
        }

    def close(self):
        self.landmarker.close()
