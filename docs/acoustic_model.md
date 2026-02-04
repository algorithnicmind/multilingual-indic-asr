# Acoustic Model

## 📋 Overview

The acoustic model maps MFCC features to phoneme probabilities. We train separate models for each language to handle different phoneme inventories.

---

## 🏗️ Architecture: CNN + BiLSTM + CTC

```
┌─────────────────────────────────────────┐
│        Input: [B × T × 39 MFCCs]        │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│     CNN Feature Extractor (3 layers)    │
│  Conv2D → ReLU → BatchNorm → MaxPool    │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│       BiLSTM Encoder (3 layers)         │
│          Hidden: 256 × 2                │
└────────────────┬────────────────────────┘
                 ▼
┌─────────────────────────────────────────┐
│        Linear → LogSoftmax              │
│     Output: [B × T × vocab_size]        │
└─────────────────────────────────────────┘
```

---

## 💻 Implementation

```python
# src/acoustic_model/model.py

import torch
import torch.nn as nn

class AcousticModel(nn.Module):
    """CNN-BiLSTM acoustic model with CTC loss."""

    def __init__(self, input_dim=39, hidden_dim=256,
                 num_layers=3, vocab_size=100, dropout=0.3):
        super().__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            (input_dim // 2) * 64,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        # Output
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: [B, T, F]
        x = x.unsqueeze(1)  # [B, 1, T, F]
        x = self.cnn(x)      # [B, C, T', F']

        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)

        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.log_softmax(x)
```

---

## 📊 Training

### Loss Function: CTC (Connectionist Temporal Classification)

CTC allows training without frame-level alignment by marginalizing over all possible alignments.

```python
loss_fn = nn.CTCLoss(blank=0, reduction='mean')

def train_step(model, batch, optimizer):
    optimizer.zero_grad()

    inputs, targets, input_lengths, target_lengths = batch
    outputs = model(inputs)

    loss = loss_fn(
        outputs.permute(1, 0, 2),
        targets,
        input_lengths,
        target_lengths
    )

    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## 🎯 Phoneme Inventories

| Language | Phonemes | Vocab Size |
| -------- | -------- | ---------- |
| English  | ~44      | 45 + blank |
| Hindi    | ~52      | 53 + blank |
| Odia     | ~48      | 49 + blank |

---

_Document Version: 1.0_
