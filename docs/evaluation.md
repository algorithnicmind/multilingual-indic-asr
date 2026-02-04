# Evaluation

## 📋 Overview

This document describes the evaluation metrics and methodology for assessing ASR system performance.

---

## 📊 Metrics

### Word Error Rate (WER)

The primary metric for ASR evaluation.

```
WER = (S + D + I) / N × 100%

Where:
- S = Substitutions
- D = Deletions
- I = Insertions
- N = Total words in reference
```

```python
def word_error_rate(reference, hypothesis):
    """Calculate WER using dynamic programming."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Edit distance matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,    # Deletion
                    d[i][j-1] + 1,    # Insertion
                    d[i-1][j-1] + 1   # Substitution
                )

    return d[-1][-1] / len(ref_words) * 100
```

### Character Error Rate (CER)

Similar to WER but at character level.

---

## 🎯 Target Results

| Metric             | English | Hindi | Odia |
| ------------------ | ------- | ----- | ---- |
| WER                | ≤40%    | ≤45%  | ≤50% |
| Language Detection | ≥90%    | ≥85%  | ≥80% |

---

## 📈 Evaluation Pipeline

```python
def evaluate_asr(model, test_set):
    """Evaluate ASR model on test set."""

    total_wer = 0
    total_cer = 0

    for audio, reference in test_set:
        hypothesis = model.transcribe(audio)

        total_wer += word_error_rate(reference, hypothesis)
        total_cer += character_error_rate(reference, hypothesis)

    return {
        'wer': total_wer / len(test_set),
        'cer': total_cer / len(test_set)
    }
```

---

_Document Version: 1.0_
