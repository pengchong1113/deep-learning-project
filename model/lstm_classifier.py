import torch
import torch.nn as nn

ACTION_LABELS = [
    'squat', 'pushup', 'situp', 'pullup',
    'jumping_jacks', 'jump_rope', 'bench_press', 'clean_and_jerk'
]

NUM_CLASSES  = len(ACTION_LABELS)
INPUT_SIZE   = 99   # 33 landmarks × 3 (x, y, z)
SEQ_LEN      = 64   # fixed sequence length (pad / truncate)


class FitnessLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])   # last layer hidden state
