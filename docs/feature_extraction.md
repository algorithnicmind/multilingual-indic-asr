# Feature Extraction

## 📋 Overview

Feature extraction transforms raw audio signals into MFCC (Mel-Frequency Cepstral Coefficients) representations suitable for speech recognition.

---

## 🎯 Why MFCC?

MFCCs are designed to:

- ✅ Mimic human auditory perception (Mel scale)
- ✅ Reduce dimensionality significantly
- ✅ Capture phonetically important information
- ✅ Be robust to noise and speaker variations
- ✅ Industry standard for 40+ years

---

## 📊 MFCC Pipeline

```
Audio → Pre-emphasis → Framing → Windowing → FFT →
Mel Filterbank → Log → DCT → MFCCs → Deltas → Normalize
```

### Step-by-Step Process

1. **Pre-emphasis** (α=0.97): `y[n] = x[n] - α·x[n-1]`
   - Boosts high frequencies to balance spectrum

2. **Framing**: 25ms frames, 10ms hop
   - Divides signal into overlapping segments

3. **Windowing**: Hamming window
   - Reduces spectral leakage at frame edges

4. **FFT**: 512-point Fast Fourier Transform
   - Converts time domain to frequency domain

5. **Mel Filterbank**: 40 triangular filters
   - Maps frequencies to perceptual Mel scale

6. **Log Compression**:
   - Mimics human loudness perception

7. **DCT**: Discrete Cosine Transform
   - Decorrelates filterbank outputs → 13 MFCCs

8. **Delta Features**: Velocity and acceleration
   - Captures temporal dynamics

9. **Normalization**: Mean/variance normalization
   - Standardizes features across utterances

---

## 🎛️ Configuration

| Parameter    | Value   | Description            |
| ------------ | ------- | ---------------------- |
| `n_mfcc`     | 13      | MFCC coefficients      |
| `n_fft`      | 512     | FFT window size        |
| `hop_length` | 160     | 10ms hop (16kHz)       |
| `win_length` | 400     | 25ms window            |
| `n_mels`     | 40      | Mel filterbank filters |
| `fmin`       | 20 Hz   | Minimum frequency      |
| `fmax`       | 8000 Hz | Maximum frequency      |

---

## 📊 Output Dimensions

**Final output**: `[T × 39]` features per audio file

- 13 MFCCs
- 13 Delta (velocity)
- 13 Delta-Delta (acceleration)

Frame rate: ~100 frames/second (10ms hop)

---

## 💻 Implementation

```python
from src.features import MFCCExtractor

config = {
    'n_mfcc': 13,
    'use_delta': True,
    'use_delta_delta': True
}

extractor = MFCCExtractor(config)
features = extractor.extract(audio)  # [T × 39]
```

---

## 📚 References

1. Davis & Mermelstein (1980) - "Comparison of parametric representations"
2. HTK Book - Chapter 5: Feature Extraction
3. Rabiner & Juang - "Fundamentals of Speech Recognition"

---

_Document Version: 1.0_
_Last Updated: February 2026_
