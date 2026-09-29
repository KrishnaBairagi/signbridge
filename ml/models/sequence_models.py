"""
ml/models/sequence_models.py

Two candidate architectures for the PRODUCTION sign recognition model
(§7/§19). Neither is trained here -- weights are random-initialized until
you run ml/training/train.py against real data (e.g. INCLUDE / INCLUDE-50,
see ml/datasets/DATASETS.md). Requires `torch` (see requirements.txt).

FEATURE_DIM matches ml/preprocessing/normalize.py's frames_to_feature_vector
output: 21*3 + 21*3 + 33*3 + 3 = 228.
"""

from __future__ import annotations

FEATURE_DIM = 228


def _lazy_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError("ml/models requires PyTorch. Install with: pip install torch") from e


def build_lstm_model(num_classes: int, hidden_size: int = 128, num_layers: int = 2):
    torch, nn = _lazy_torch()

    class LSTMSignClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=FEATURE_DIM, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, num_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, FEATURE_DIM)
            output, (h_n, _) = self.lstm(x)
            # concat last layer's forward/backward hidden states
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
            return self.classifier(final_hidden)

    return LSTMSignClassifier()


def build_transformer_model(num_classes: int, d_model: int = 128, nhead: int = 4,
                             num_layers: int = 3, max_seq_len: int = 64):
    torch, nn = _lazy_torch()

    class TransformerSignClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(FEATURE_DIM, d_model)
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=0.2, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_model // 2, num_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, FEATURE_DIM)
            seq_len = x.shape[1]
            h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
            h = self.encoder(h)
            pooled = h.mean(dim=1)  # mean pooling over time
            return self.classifier(pooled)

    return TransformerSignClassifier()
