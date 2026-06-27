"""
Generate training plots from the logged training history.
Run: python scripts/plot_from_log.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ── Training history from the original run ────────────────────────────────────
train_loss = [2.0348,1.8586,1.7583,1.7254,1.6655,1.5944,1.5484,1.4445,1.3620,1.2133,
              1.3847,1.1904,1.2065,1.1551,1.1914,1.0730,1.5631,1.3333,1.1292,1.0291,
              0.9670,0.8912,0.9096,1.0129,0.9122,0.8586,0.8269,0.7823,0.7724,0.8390,
              0.7191,0.7795,0.7001,0.9762,0.8227,0.7687,0.7369,0.6511,0.6195,0.5943,
              0.6195,0.5925,0.5173,0.7492,0.6512,0.6230,0.4866,0.4569,0.4533,0.4595,
              0.5626,0.5317,0.6324,0.5567,0.5048,0.4968,0.5358,0.4790,0.4746,0.4332]

val_loss   = [1.9408,1.8133,1.7252,1.7036,1.6375,1.6058,1.5176,1.4945,1.4884,1.5306,
              1.3262,1.3540,1.3227,1.2806,1.1434,1.2516,1.6701,1.2516,1.1203,1.0926,
              1.0800,0.9781,0.9651,0.9977,0.9763,1.0094,0.9076,0.8836,0.8606,0.8589,
              0.8527,0.9075,1.1136,1.0908,0.9651,0.9861,0.8715,0.8439,0.8232,0.8444,
              0.8160,0.7674,0.8487,0.8404,0.8694,0.8104,0.7996,0.7849,0.7953,0.8767,
              0.7658,0.9544,0.9767,0.7619,0.8823,0.8057,0.7615,0.7476,0.7137,0.7872]

train_acc  = [0.202,0.310,0.314,0.322,0.364,0.376,0.397,0.423,0.514,0.563,
              0.479,0.592,0.559,0.589,0.598,0.632,0.495,0.500,0.615,0.648,
              0.653,0.690,0.726,0.666,0.697,0.706,0.725,0.737,0.749,0.737,
              0.775,0.749,0.782,0.653,0.714,0.754,0.751,0.810,0.787,0.775,
              0.787,0.798,0.829,0.760,0.789,0.798,0.822,0.850,0.862,0.848,
              0.826,0.828,0.770,0.791,0.798,0.826,0.815,0.841,0.841,0.861]

val_acc    = [0.314,0.289,0.323,0.324,0.348,0.382,0.387,0.435,0.457,0.435,
              0.518,0.537,0.548,0.553,0.598,0.576,0.377,0.567,0.620,0.603,
              0.616,0.701,0.698,0.677,0.701,0.684,0.713,0.716,0.747,0.737,
              0.737,0.699,0.615,0.640,0.703,0.725,0.720,0.749,0.752,0.749,
              0.754,0.762,0.754,0.762,0.723,0.759,0.757,0.778,0.784,0.764,
              0.783,0.715,0.696,0.750,0.761,0.774,0.744,0.791,0.805,0.796]

# ── Per-class accuracy ────────────────────────────────────────────────────────
per_class = {
    'squat':          (87,  117),
    'pushup':         (91,  107),
    'situp':          (33,   50),
    'pullup':         (89,  101),
    'jumping_jacks':  (55,   56),
    'jump_rope':      (33,   42),
    'bench_press':    (47,   71),
    'clean_and_jerk': (39,   45),
}

# ── Plot settings ─────────────────────────────────────────────────────────────
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)
plt.style.use('dark_background')
epochs = range(1, 61)
best_epoch = 59   # epoch where val_acc peaked at 80.5%

# ── Plot 1: Loss curve ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, label='Train Loss', color='#00d4ff', linewidth=1.5)
ax.plot(epochs, val_loss,   label='Val Loss',   color='#7b2ff7', linewidth=1.5)
ax.axvline(x=best_epoch, color='white', linestyle='--', alpha=0.4, label=f'Best epoch ({best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(results_dir / 'loss_curve.png', dpi=150)
plt.close(fig)
print('Saved: loss_curve.png')

# ── Plot 2: Accuracy curve ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, [a*100 for a in train_acc], label='Train Acc', color='#00d4ff', linewidth=1.5)
ax.plot(epochs, [a*100 for a in val_acc],   label='Val Acc',   color='#7b2ff7', linewidth=1.5)
ax.axvline(x=best_epoch, color='white', linestyle='--', alpha=0.4, label=f'Best epoch ({best_epoch})')
ax.axhline(y=80.5, color='#ffcc00', linestyle=':', alpha=0.6, label='Best val acc 80.5%')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Training & Validation Accuracy')
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(results_dir / 'accuracy_curve.png', dpi=150)
plt.close(fig)
print('Saved: accuracy_curve.png')

# ── Plot 3: Per-class accuracy ────────────────────────────────────────────────
names  = list(per_class.keys())
accs   = [per_class[n][0] / per_class[n][1] * 100 for n in names]
counts = [f'{per_class[n][0]}/{per_class[n][1]}' for n in names]
labels = [n.replace('_', '\n') for n in names]
colors = ['#00d4ff' if a >= 80 else '#7b2ff7' if a >= 65 else '#ff4f4f' for a in accs]
mean_acc = np.mean(accs)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, accs, color=colors, width=0.6)
for bar, acc, cnt in zip(bars, accs, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{acc:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=8.5)
ax.axhline(y=mean_acc, color='white', linestyle='--', alpha=0.5, label=f'Overall {mean_acc:.1f}%')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 115)
ax.set_title('Per-class Accuracy on Test Set  (Best model, Epoch 59)')
ax.legend()
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(results_dir / 'per_class_accuracy.png', dpi=150)
plt.close(fig)
print('Saved: per_class_accuracy.png')

print(f'\nAll plots saved to {results_dir}/')
