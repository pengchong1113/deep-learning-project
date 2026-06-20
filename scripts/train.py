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
            kp    = data['keypoints'].astype(np.float32)   # (frames, 33, 3)
            kp    = kp.reshape(len(kp), -1)                # (frames, 99)
            kp    = self._pad_or_truncate(kp)              # (SEQ_LEN, 99)
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


# ── Training helpers ───────────────────────────────────────────────────────────

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


def evaluate_per_class(model, loader, device):
    model.eval()
    per_class = {name: {'correct': 0, 'total': 0} for name in ACTION_LABELS}

    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            preds   = model(x).argmax(1)
            for pred, true in zip(preds, y):
                name = ACTION_LABELS[true.item()]
                per_class[name]['total']   += 1
                per_class[name]['correct'] += (pred == true).item()

    log.info('--- Per-class accuracy ---')
    for name, v in per_class.items():
        acc = v['correct'] / v['total'] if v['total'] > 0 else 0
        log.info(f'  {name:<20} {acc:.1%}  ({v["correct"]}/{v["total"]})')


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/processed/keypoints')
    parser.add_argument('--model_dir',  default='model/saved_model')
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--hidden',     type=int,   default=128)
    parser.add_argument('--layers',     type=int,   default=2)
    parser.add_argument('--dropout',    type=float, default=0.3)
    return parser.parse_args()


def main():
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    # Datasets
    train_ds = KeypointDataset(data_dir, split='train')
    test_ds  = KeypointDataset(data_dir, split='test')
    log.info(f'Train: {len(train_ds)} samples | Test: {len(test_ds)} samples')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
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

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, test_loader,  criterion, optimizer, device, training=False)
        scheduler.step(val_loss)

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
    log.info('Saved to model/saved_model/best_model.pt')

    # Load best and evaluate per class
    ckpt = torch.load(model_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    evaluate_per_class(model, test_loader, device)


if __name__ == '__main__':
    main()
