# Language Model

## 📋 Overview

The language model predicts valid word sequences, improving transcription accuracy by incorporating linguistic knowledge.

---

## 🎯 Purpose

- Assign probabilities to word sequences
- Disambiguate acoustically similar words
- Improve fluency of output text

---

## 🏗️ N-gram Language Model

### Architecture

```
Trigram Model: P(w₃ | w₁, w₂)

Storage:
├── Unigrams: {word: count}
├── Bigrams:  {(w1, w2): count}
└── Trigrams: {(w1, w2, w3): count}
```

---

## 💻 Implementation

```python
# src/language_model/ngram.py

import pickle
from collections import defaultdict

class NGramLanguageModel:
    """N-gram language model with Kneser-Ney smoothing."""

    def __init__(self, n=3, discount=0.75):
        self.n = n
        self.discount = discount
        self.ngram_counts = [defaultdict(int) for _ in range(n)]
        self.vocab = set()

    def train(self, corpus):
        """Train on text corpus."""
        for sentence in corpus:
            tokens = ['<s>'] * (self.n - 1) + sentence.split() + ['</s>']
            self.vocab.update(tokens)

            for i in range(len(tokens) - self.n + 1):
                for j in range(self.n):
                    ngram = tuple(tokens[i:i+j+1])
                    self.ngram_counts[j][ngram] += 1

    def probability(self, word, context):
        """Calculate P(word | context) with smoothing."""
        ngram = context + (word,)
        count = self.ngram_counts[len(ngram)-1][ngram]
        context_count = self.ngram_counts[len(context)-1][context] if context else sum(self.ngram_counts[0].values())

        if context_count == 0:
            return 1 / len(self.vocab)

        # Kneser-Ney smoothing
        prob = max(count - self.discount, 0) / context_count
        lambda_weight = self.discount * len(self.ngram_counts[len(context)]) / context_count

        if len(context) > 0:
            prob += lambda_weight * self.probability(word, context[1:])
        else:
            prob += lambda_weight / len(self.vocab)

        return prob

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
```

---

## 📊 Training

### Text Preprocessing

```python
def preprocess_text(text, language):
    """Preprocess text for language model training."""
    text = text.lower()

    if language == 'english':
        text = re.sub(r'[^a-z\s]', '', text)
    elif language == 'hindi':
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    elif language == 'odia':
        text = re.sub(r'[^\u0B00-\u0B7F\s]', '', text)

    return ' '.join(text.split())
```

---

## 📈 Statistics

| Language | Vocab Size | Bigrams | Trigrams  |
| -------- | ---------- | ------- | --------- |
| English  | 30,000     | 500,000 | 1,500,000 |
| Hindi    | 25,000     | 400,000 | 1,200,000 |
| Odia     | 15,000     | 200,000 | 600,000   |

---

_Document Version: 1.0_
