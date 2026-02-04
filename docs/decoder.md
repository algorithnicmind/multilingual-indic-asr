# Decoder

## 📋 Overview

The decoder combines acoustic model output with language model probabilities to produce the final transcription.

---

## 🎯 Objective

- Input: Acoustic model log probabilities [T × V]
- Input: Language model
- Output: Best text hypothesis

---

## 🏗️ Algorithms

### 1. Greedy Decoding

```python
def greedy_decode(log_probs, vocab):
    """Simple greedy decoding - take best at each step."""
    indices = log_probs.argmax(dim=-1)

    # Remove blanks and repeated
    decoded = []
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev:  # 0 = blank
            decoded.append(vocab[idx])
        prev = idx

    return ''.join(decoded)
```

### 2. Beam Search

```python
# src/decoder/beam_search.py

class BeamSearchDecoder:
    """Beam search with language model integration."""

    def __init__(self, vocab, lm, beam_width=10, lm_weight=0.4):
        self.vocab = vocab
        self.lm = lm
        self.beam_width = beam_width
        self.lm_weight = lm_weight

    def decode(self, log_probs):
        """
        Beam search decoding.

        Args:
            log_probs: [T × V] tensor

        Returns:
            Best hypothesis string
        """
        T, V = log_probs.shape
        beams = [{'prefix': '', 'score': 0.0}]

        for t in range(T):
            candidates = []

            for beam in beams:
                for v in range(V):
                    if v == 0:  # blank
                        new_prefix = beam['prefix']
                    else:
                        char = self.vocab[v]
                        if beam['prefix'] and beam['prefix'][-1] == char:
                            continue
                        new_prefix = beam['prefix'] + char

                    # Acoustic score
                    am_score = log_probs[t, v].item()

                    # Language model score
                    lm_score = self._get_lm_score(new_prefix)

                    # Combined score
                    score = beam['score'] + am_score + self.lm_weight * lm_score

                    candidates.append({
                        'prefix': new_prefix,
                        'score': score
                    })

            # Keep top beams
            beams = sorted(candidates, key=lambda x: x['score'], reverse=True)[:self.beam_width]

        return beams[0]['prefix']

    def _get_lm_score(self, prefix):
        if len(prefix) < 2:
            return 0.0
        words = prefix.split()
        if len(words) < 2:
            return 0.0
        return np.log(self.lm.probability(words[-1], tuple(words[-3:-1])) + 1e-10)
```

---

## 📊 Comparison

| Method      | Quality | Speed  |
| ----------- | ------- | ------ |
| Greedy      | Lower   | Fast   |
| Beam (w=5)  | Medium  | Medium |
| Beam (w=10) | Best    | Slower |

---

_Document Version: 1.0_
