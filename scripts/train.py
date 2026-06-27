"""
Train the FitnessLSTM classifier on extracted Penn Action keypoints.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 80 --batch_size 32 --lr 1e-3
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent.parent))
from model.lstm_classifier import FitnessLSTM, ACTION_LABELS, SEQ_LEN

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.samples = []
        for f in sorted(data_dir.glob('*.npz')):
            data = np.load(str(f), allow_pickle=True)
            if str(data['split']) != split:
                continue
            kp    = data['keypoints'].astype(np.float32)
            kp    = kp.reshape(len(kp), -1)
            kp    = self._pad_or_truncate(kp)
            label = int(data['label'])
            self.samples.append((kp, label))

    def _pad_or_truncate(self, kp: np.ndarray) -> np.ndarray:
        n = len(kp)
        if n >= SEQ_LEN:
            return kp[:SEQ_LEN]
        pad = np.zeros((SEQ_LEN - n, kp.shape[1]), dtype=np.float32)
        return np.concatenate([kp, pad], axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp, label = self.samples[idx]
        return torch.tensor(kp), torch.tensor(label)


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def evaluate_per_class(model, loader, device) -> dict:
    model.eval()
    per_class = {name: {'correct': 0, 'total': 0} for name in ACTION_LABELS}

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for pred, true in zip(preds, y):
                name = ACTION_LABELS[true.item()]
                per_class[name]['total']   += 1
                per_class[name]['correct'] += (pred == true).item()

    log.info('--- Per-class accuracy ---')
    for name, v in per_class.items():
        acc = v['correct'] / v['total'] if v['total'] > 0 else 0
        log.info(f'  {name:<20} {acc:.1%}  ({v["correct"]}/{v["total"]})')

    return per_class


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_plots(history: dict, per_class: dict, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('dark_background')

    # ── Plot 1: Loss curve ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#00d4ff')
    ax.plot(epochs, history['val_loss'],   label='Val Loss',   color='#7b2ff7')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'loss_curve.png', dpi=150)
    plt.close(fig)

    # ── Plot 2: Accuracy curve ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Acc', color='#00d4ff')
    ax.plot(epochs, [a * 100 for a in history['val_acc']],   label='Val Acc',   color='#7b2ff7')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'accuracy_curve.png', dpi=150)
    plt.close(fig)

    # ── Plot 3: Per-class accuracy bar chart ──────────────────────────────────
    names = list(per_class.keys())
    accs  = [
        per_class[n]['correct'] / per_class[n]['total'] * 100
        if per_class[n]['total'] > 0 else 0
        for n in names
    ]
    labels = [n.replace('_', '\n') for n in names]
    colors = ['#00d4ff' if a >= 80 else '#7b2ff7' if a >= 65 else '#ff4f4f' for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, color=colors, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.set_title('Per-class Accuracy on Test Set')
    ax.axhline(y=sum(accs) / len(accs), color='white', linestyle='--', alpha=0.4, label=f'Mean {sum(accs)/len(accs):.1f}%')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / 'per_class_accuracy.png', dpi=150)
    plt.close(fig)

    log.info(f'Plots saved to {results_dir}/')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default='data/processed/keypoints')
    parser.add_argument('--model_dir',   default='model/saved_model')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--epochs',      type=int,   default=60)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--hidden',      type=int,   default=128)
    parser.add_argument('--layers',      type=int,   default=2)
    parser.add_argument('--dropout',     type=float, default=0.3)
    return parser.parse_args()


def main():
    args        = parse_args()
    data_dir    = Path(args.data_dir)
    model_dir   = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    train_ds = KeypointDataset(data_dir, split='train')
    test_ds  = KeypointDataset(data_dir, split='test')
    log.info(f'Train: {len(train_ds)} samples | Test: {len(test_ds)} samples')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FitnessLSTM(
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    log.info(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, test_loader,  criterion, optimizer, device, training=False)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        log.info(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'train loss {train_loss:.4f} acc {train_acc:.1%} | '
            f'val loss {val_loss:.4f} acc {val_acc:.1%}'
            + (' ← best' if val_acc > best_val_acc else '')
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'args':       vars(args),
            }, model_dir / 'best_model.pt')

    log.info(f'\nBest val accuracy: {best_val_acc:.1%}')

    ckpt = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    per_class = evaluate_per_class(model, test_loader, device)

    save_plots(history, per_class, results_dir)


if __name__ == '__main__':
    main()
