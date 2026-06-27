"""
Rep counting via joint angle analysis.
Detects periodic motion in the keypoint sequence and counts repetitions.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

# MediaPipe landmark indices
LM = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow':    13, 'right_elbow':    14,
    'left_wrist':    15, 'right_wrist':    16,
    'left_hip':      23, 'right_hip':      24,
    'left_knee':     25, 'right_knee':     26,
    'left_ankle':    27, 'right_ankle':    28,
}

# Per action: which three joints form the angle, and count peaks or valleys
ACTION_CONFIG = {
    'squat':          {'joints': ('left_hip',      'left_knee',  'left_ankle'),    'count': 'valleys'},
    'pushup':         {'joints': ('left_shoulder', 'left_elbow', 'left_wrist'),    'count': 'valleys'},
    'situp':          {'joints': ('left_shoulder', 'left_hip',   'left_knee'),     'count': 'valleys'},
    'pullup':         {'joints': ('left_shoulder', 'left_elbow', 'left_wrist'),    'count': 'valleys'},
    'jumping_jacks':  {'joints': ('left_hip',      'left_shoulder', 'left_wrist'), 'count': 'peaks'},
    'jump_rope':      {'joints': ('left_hip',      'left_knee',  'left_ankle'),    'count': 'peaks'},
    'bench_press':    {'joints': ('left_shoulder', 'left_elbow', 'left_wrist'),    'count': 'valleys'},
    'clean_and_jerk': {'joints': ('left_shoulder', 'left_elbow', 'left_wrist'),    'count': 'peaks'},
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b formed by the vectors b→a and b→c (degrees)."""
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _angle_sequence(keypoints: np.ndarray, j1: int, j2: int, j3: int) -> np.ndarray:
    """
    Compute joint angle at j2 for every frame.
    keypoints: (frames, 33, 3)
    """
    return np.array([
        _angle(kp[j1, :2], kp[j2, :2], kp[j3, :2])
        for kp in keypoints
    ])


def _smooth(signal: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing — removes jitter while preserving peaks."""
    n = len(signal)
    if n < 7:
        return signal
    window = min(n if n % 2 == 1 else n - 1, 11)
    return savgol_filter(signal, window_length=window, polyorder=2)


def count_reps(keypoints: np.ndarray, action: str) -> dict:
    """
    Count repetitions for a given action from a keypoint sequence.

    Args:
        keypoints: (frames, 33, 3) — MediaPipe keypoints, NOT padded/truncated
        action:    one of ACTION_CONFIG keys

    Returns:
        dict with 'reps', 'angle_sequence', and 'peaks' for visualization
    """
    config = ACTION_CONFIG.get(action)
    if config is None:
        return {'reps': 0, 'angle_sequence': [], 'peaks': []}

    j1 = LM[config['joints'][0]]
    j2 = LM[config['joints'][1]]
    j3 = LM[config['joints'][2]]

    angles  = _angle_sequence(keypoints, j1, j2, j3)
    smoothed = _smooth(angles)

    signal_range = smoothed.max() - smoothed.min()

    # If total range of motion < 20 degrees, no meaningful reps
    if signal_range < 20.0:
        return {'reps': 0, 'angle_sequence': smoothed.tolist(), 'peaks': []}

    # Prominence: per-action factor (default 35%) of signal range
    prominence = signal_range * config.get('prominence_factor', 0.35)
    # Minimum frames between reps: video length / 8, at least 8 frames
    min_dist = max(len(smoothed) // 8, 8)

    if config['count'] == 'valleys':
        peaks, _ = find_peaks(-smoothed, prominence=prominence, distance=min_dist)
    else:
        peaks, _ = find_peaks(smoothed,  prominence=prominence, distance=min_dist)

    return {
        'reps':           int(len(peaks)),
        'angle_sequence': smoothed.tolist(),
        'peaks':          peaks.tolist(),
    }
