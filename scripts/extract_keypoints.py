"""
Extract MediaPipe pose keypoints from Penn Action Dataset.

Usage:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --data_dir data/raw/Penn_Action --output_dir data/processed/keypoints
"""

import argparse
import logging
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import scipy.io as sio

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

FITNESS_ACTIONS = [
    'squat', 'pushup', 'situp', 'pullup',
    'jumping_jacks', 'jump_rope', 'bench_press', 'clean_and_jerk'
]
ACTION_TO_IDX = {action: idx for idx, action in enumerate(FITNESS_ACTIONS)}

MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
MODEL_PATH = Path('models/pose_landmarker_lite.task')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/raw/Penn_Action')
    parser.add_argument('--output_dir', default='data/processed/keypoints')
    return parser.parse_args()


def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f'Downloading pose landmarker model (~4MB)...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info(f'Model saved to {MODEL_PATH}')


def build_landmarker():
    ensure_model()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def load_fitness_videos(labels_dir: Path) -> list:
    videos = []
    for mat_path in sorted(labels_dir.glob('*.mat')):
        mat    = sio.loadmat(str(mat_path))
        action = str(mat['action'][0])
        if action not in FITNESS_ACTIONS:
            continue
        videos.append({
            'video_id': mat_path.stem,
            'action':   action,
            'label':    ACTION_TO_IDX[action],
            'split':    'train' if int(mat['train'][0][0]) == 1 else 'test',
        })
    return videos


def extract_keypoints(video_id: str, frames_dir: Path, landmarker) -> Optional[np.ndarray]:
    """
    Returns (num_frames, 33, 3) array of x, y, z keypoints.
    Frames with no detected pose are filled with zeros.
    """
    frame_paths = sorted((frames_dir / video_id).glob('*.jpg'))
    if not frame_paths:
        return None

    seq = []
    for frame_path in frame_paths:
        img     = cv2.imread(str(frame_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result  = landmarker.detect(mp_img)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            kp  = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        else:
            kp = np.zeros((33, 3), dtype=np.float32)

        seq.append(kp)

    return np.array(seq)  # (num_frames, 33, 3)


def print_summary(output_dir: Path):
    saved   = list(output_dir.glob('*.npz'))
    counter = Counter()
    for f in saved:
        data = np.load(str(f), allow_pickle=True)
        counter[str(data['action'])] += 1

    log.info('--- Extraction Summary ---')
    log.info(f'{"Action":<20} {"Saved":>6}')
    for action in FITNESS_ACTIONS:
        log.info(f'{action:<20} {counter[action]:>6}')
    log.info(f'{"TOTAL":<20} {sum(counter.values()):>6}')


def main():
    args       = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = data_dir / 'labels'
    frames_dir = data_dir / 'frames'

    videos = load_fitness_videos(labels_dir)
    log.info(f'Found {len(videos)} fitness videos')

    skipped = []

    total = len(videos)
    with build_landmarker() as landmarker:
        for i, video in enumerate(videos):
            out_path = output_dir / f"{video['video_id']}.npz"
            if out_path.exists():
                continue

            keypoints = extract_keypoints(video['video_id'], frames_dir, landmarker)
            if keypoints is None:
                skipped.append(video['video_id'])
                continue

            np.savez(
                str(out_path),
                keypoints=keypoints,
                label=video['label'],
                action=video['action'],
                split=video['split'],
            )

            if (i + 1) % 50 == 0 or (i + 1) == total:
                done = len(list(output_dir.glob('*.npz')))
                log.info(f'Progress: {i+1}/{total} processed | saved: {done}')

    if skipped:
        log.warning(f'Skipped {len(skipped)} videos (no frames): {skipped}')

    print_summary(output_dir)


if __name__ == '__main__':
    main()
