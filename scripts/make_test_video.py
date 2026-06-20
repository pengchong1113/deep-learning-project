"""
Convert Penn Action frame sequence into a .mp4 test video.

Usage:
    python scripts/make_test_video.py               # picks one video per action
    python scripts/make_test_video.py --video_id 0001  # specific video
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

FRAMES_DIR = Path('data/raw/Penn_Action/frames')
LABELS_DIR = Path('data/raw/Penn_Action/labels')
OUTPUT_DIR = Path('data/test_videos')

FITNESS_ACTIONS = [
    'squat', 'pushup', 'situp', 'pullup',
    'jumping_jacks', 'jump_rope', 'bench_press', 'clean_and_jerk'
]


def frames_to_video(video_id: str, output_path: Path, fps: int = 15):
    frame_paths = sorted((FRAMES_DIR / video_id).glob('*.jpg'))
    if not frame_paths:
        print(f'No frames found for {video_id}')
        return

    first = cv2.imread(str(frame_paths[0]))
    h, w  = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h),
    )
    for fp in frame_paths:
        writer.write(cv2.imread(str(fp)))
    writer.release()
    print(f'Saved: {output_path}  ({len(frame_paths)} frames, {len(frame_paths)/fps:.1f}s)')


def pick_one_per_action() -> dict:
    by_action: dict = {a: [] for a in FITNESS_ACTIONS}
    for mat_path in LABELS_DIR.glob('*.mat'):
        mat    = sio.loadmat(str(mat_path))
        action = str(mat['action'][0])
        if action in FITNESS_ACTIONS:
            by_action[action].append(mat_path.stem)
    return {a: random.choice(ids) for a, ids in by_action.items() if ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', default=None, help='specific Penn Action video ID, e.g. 0001')
    parser.add_argument('--fps',      type=int, default=15)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.video_id:
        mat    = sio.loadmat(str(LABELS_DIR / f'{args.video_id}.mat'))
        action = str(mat['action'][0])
        out    = OUTPUT_DIR / f'{action}_{args.video_id}.mp4'
        frames_to_video(args.video_id, out, fps=args.fps)
    else:
        picks = pick_one_per_action()
        for action, video_id in picks.items():
            out = OUTPUT_DIR / f'{action}_{video_id}.mp4'
            frames_to_video(video_id, out, fps=args.fps)
        print(f'\nTest videos saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
