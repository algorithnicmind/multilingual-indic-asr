# MFCC Feature Extraction

## 📋 Overview

Feature extraction transforms raw audio signals into MFCC (Mel-Frequency Cepstral Coefficients) representations suitable for speech recognition.

---

## 🎯 Why MFCC?

- Mimics human auditory perception (Mel scale)
- Reduces dimensionality significantly
- Captures phonetically important information
- Standard in ASR for decades

---

## 📊 Pipeline

```
Audio → Pre-emphasis → Framing → Windowing → FFT →
Mel Filterbank → Log → DCT → MFCCs → Deltas → Normalize
```

---

## 🎛️ Parameters

| Parameter  | Value   | Description         |
| ---------- | ------- | ------------------- |
| n_mfcc     | 13      | Number of MFCCs     |
| n_fft      | 512     | FFT window size     |
| hop_length | 160     | 10ms hop (at 16kHz) |
| win_length | 400     | 25ms window         |
| n_mels     | 40      | Mel filters         |
| fmin       | 20 Hz   | Min frequency       |
| fmax       | 8000 Hz | Max frequency       |

---

## 📊 Output

- **13 MFCCs** + **13 Deltas** + **13 Delta-Deltas** = **39 features/frame**
- Frame rate: ~100 frames/second

---

_See full implementation in [docs/feature_extraction.md](feature_extraction.md)_
