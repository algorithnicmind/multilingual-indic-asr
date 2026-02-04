# Language Identification

## 📋 Overview

The Language Identification (LID) module automatically detects whether the input audio is in English, Hindi, or Odia. This enables routing to the appropriate language-specific ASR model.

---

## 🎯 Objective

- **Input**: MFCC features from audio
- **Output**: Language label (`en`, `hi`, `or`)
- **Target Accuracy**: ≥ 85%

---

## 🏗️ Architecture Options

### Option A: SVM Classifier (Baseline)

```
MFCC Features [T × 39]
        │
        ▼ Temporal Aggregation
   [Mean, Std] → [78 features]
        │
        ▼ SVM (RBF kernel)
   Language Prediction
```

### Option B: Neural Network

```
MFCC Features [T × 39]
        │
        ▼ LSTM (128 hidden)
   Hidden State [128]
        │
        ▼ Dense + Softmax
   Language Probs [3]
```

---

## 💻 Implementation

```python
# src/language_id/model.py

import numpy as np
from sklearn.svm import SVC
import torch
import torch.nn as nn

class SVMLanguageIdentifier:
    """SVM-based language identification."""

    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)
        self.labels = ['english', 'hindi', 'odia']

    def _aggregate_features(self, features):
        """Aggregate temporal features."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return np.concatenate([mean, std])

    def fit(self, X, y):
        """Train the classifier."""
        X_agg = np.array([self._aggregate_features(f) for f in X])
        self.model.fit(X_agg, y)

    def predict(self, features):
        """Predict language."""
        X_agg = self._aggregate_features(features).reshape(1, -1)
        return self.labels[self.model.predict(X_agg)[0]]


class NeuralLanguageIdentifier(nn.Module):
    """Neural network language identifier."""

    def __init__(self, input_dim=39, hidden_dim=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden.squeeze(0))
```

---

## 📊 Training

### Data Preparation

- Balance samples across languages
- Use 80/10/10 train/val/test split
- Extract MFCC features from all samples

### Training Script

```python
# src/language_id/train.py

def train_lid_model(train_data, val_data):
    model = SVMLanguageIdentifier()
    X_train, y_train = zip(*train_data)
    model.fit(list(X_train), list(y_train))

    # Evaluate
    accuracy = evaluate(model, val_data)
    print(f"Validation Accuracy: {accuracy:.2%}")

    return model
```

---

## 📈 Expected Results

| Model  | Accuracy | Latency |
| ------ | -------- | ------- |
| SVM    | 85-90%   | <50ms   |
| Neural | 88-92%   | <100ms  |

---

_Document Version: 1.0_
