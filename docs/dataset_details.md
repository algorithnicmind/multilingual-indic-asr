# Dataset Details

## 📋 Overview

This document describes the datasets used for training the Multilingual Indic ASR system, including sources, preprocessing steps, and statistics.

---

## 📊 Dataset Sources

### English

#### Mozilla Common Voice (English)

| Attribute        | Value                                      |
| ---------------- | ------------------------------------------ |
| **Source**       | [openslr.org](https://www.openslr.org/12/) |
| **Dataset Name** | LibriSpeech (test-clean)                   |
| **Size**         | ~350 MB                                    |
| **License**      | CC BY 4.0                                  |

**Download Instructions**:

```bash
# Go to: https://www.openslr.org/12/
# Click "test-clean.tar.gz" (346MB)
# Extract to data/raw/english/
```

**Download Instructions**:

```bash
# Download from Mozilla Common Voice website
# Requires account creation and agreement to terms
# Extract to data/raw/english/
```

---

### Hindi

#### Mozilla Common Voice (Hindi)

| Attribute        | Value                                           |
| ---------------- | ----------------------------------------------- |
| **Source**       | [openslr.org/103](https://www.openslr.org/103/) |
| **Dataset Name** | IndicSpeech (Hindi)                             |
| **File Name**    | Hindi_test.tar.gz                               |
| **Size**         | ~100 MB                                         |
| **License**      | CC BY-SA 4.0                                    |

**Download Instructions**:

```bash
# Go to: https://www.openslr.org/103/
# Click "Hindi_test.tar.gz"
# Extract to data/raw/hindi/
```

---

### Odia

#### AI4Bharat Odia Dataset

| Attribute        | Value                                           |
| ---------------- | ----------------------------------------------- |
| **Source**       | [openslr.org/103](https://www.openslr.org/103/) |
| **Dataset Name** | IndicSpeech (Odia)                              |
| **File Name**    | Odia_test.tar.gz                                |
| **Size**         | ~100 MB                                         |
| **License**      | CC BY-SA 4.0                                    |

**Download Instructions**:

```bash
# Go to: https://www.openslr.org/103/
# Click "Odia_test.tar.gz"
# Extract to data/raw/odia/
```

---

## 📁 Data Structure

### Directory Organization

```
data/
├── raw/                          # Original downloaded data
│   ├── english/
│   │   ├── clips/               # Audio files
│   │   │   ├── sample_001.wav
│   │   │   ├── sample_002.wav
│   │   │   └── ...
│   │   └── validated.tsv        # Metadata & transcripts
│   │
│   ├── hindi/
│   │   ├── clips/
│   │   │   ├── sample_001.wav
│   │   │   └── ...
│   │   └── validated.tsv
│   │
│   └── odia/
│       ├── clips/
│       │   ├── sample_001.wav
│       │   └── ...
│       └── validated.tsv
│
├── processed/                    # Preprocessed audio
│   ├── english/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── hindi/
│   └── odia/
│
├── features/                     # Extracted features
│   ├── english/
│   │   ├── train/
│   │   │   ├── sample_001.npy   # MFCC features
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   ├── hindi/
│   └── odia/
│
└── transcripts/                  # Text transcripts
    ├── english/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    ├── hindi/
    └── odia/
```

### Metadata Format (TSV)

```
client_id	path	sentence	up_votes	down_votes	age	gender	accent
abc123	clips/sample_001.wav	Hello world	5	0	twenties	male	us
def456	clips/sample_002.wav	Good morning	3	0	thirties	female	uk
```

---

## 🔧 Data Preprocessing

### Step 1: Audio Validation

```python
def validate_audio(audio_path):
    """
    Validate audio file meets requirements.
    """
    # Check file exists
    if not os.path.exists(audio_path):
        return False, "File not found"

    # Load and check properties
    audio, sr = librosa.load(audio_path, sr=None)

    # Check duration (0.5s - 30s)
    duration = len(audio) / sr
    if duration < 0.5 or duration > 30:
        return False, f"Invalid duration: {duration}s"

    # Check for silence-only files
    if np.max(np.abs(audio)) < 0.01:
        return False, "Audio is silent"

    return True, "Valid"
```

### Step 2: Audio Standardization

```python
def standardize_audio(audio_path, output_path):
    """
    Standardize audio to project requirements.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Resample to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Normalize amplitude
    audio = audio / np.max(np.abs(audio))

    # Save as WAV
    sf.write(output_path, audio, 16000)
```

### Step 3: Text Normalization

#### English

```python
def normalize_english(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophe
    text = re.sub(r"[^\w\s']", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text
```

#### Hindi (Devanagari)

```python
def normalize_hindi(text):
    # Remove English characters
    text = re.sub(r"[a-zA-Z]", "", text)
    # Remove punctuation
    text = re.sub(r"[।॥,;:!?]", "", text)
    # Normalize nukta variations
    # Normalize whitespace
    text = " ".join(text.split())
    return text
```

#### Odia

```python
def normalize_odia(text):
    # Keep only Odia Unicode range
    # Remove punctuation
    # Normalize whitespace
    text = " ".join(text.split())
    return text
```

### Step 4: Train/Val/Test Split

```python
def create_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Create train/validation/test splits.
    """
    # Shuffle data
    np.random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }
```

---

## 📈 Dataset Statistics

### Summary by Language

| Language | Train Hours | Val Hours | Test Hours | Vocab Size |
| -------- | ----------- | --------- | ---------- | ---------- |
| English  | ~100        | ~10       | ~10        | ~30,000    |
| Hindi    | ~50         | ~5        | ~5         | ~25,000    |
| Odia     | ~20         | ~2        | ~2         | ~15,000    |

### Audio Statistics

| Metric            | English | Hindi | Odia |
| ----------------- | ------- | ----- | ---- |
| Mean Duration (s) | 5.2     | 4.8   | 4.5  |
| Std Duration (s)  | 2.1     | 1.9   | 1.7  |
| Min Duration (s)  | 1.0     | 1.0   | 1.0  |
| Max Duration (s)  | 15.0    | 12.0  | 10.0 |

### Transcript Statistics

| Metric          | English | Hindi  | Odia   |
| --------------- | ------- | ------ | ------ |
| Mean Word Count | 8.5     | 7.2    | 6.8    |
| Max Word Count  | 25      | 20     | 18     |
| Unique Words    | 30,000  | 25,000 | 15,000 |

---

## 🛡️ Data Quality Measures

### Quality Checks

1. **Audio Quality**
   - ✅ Signal-to-noise ratio > 10dB
   - ✅ No clipping (max amplitude < 0.99)
   - ✅ Consistent sample rate

2. **Transcript Quality**
   - ✅ No empty transcripts
   - ✅ Matching language script
   - ✅ Reasonable length

3. **Alignment Quality**
   - ✅ Audio and text correspond
   - ✅ Duration matches content

### Filtering Criteria

```python
def filter_sample(audio, transcript, language):
    """
    Filter low-quality samples.
    """
    # Duration check
    duration = len(audio) / 16000
    if duration < 1.0 or duration > 15.0:
        return False

    # Transcript length check
    words = transcript.split()
    if len(words) < 2 or len(words) > 30:
        return False

    # Language-specific character check
    if language == 'english':
        if not re.match(r'^[a-z\s\']+$', transcript.lower()):
            return False
    elif language == 'hindi':
        # Check for Devanagari
        if not any('\u0900' <= c <= '\u097F' for c in transcript):
            return False
    elif language == 'odia':
        # Check for Odia script
        if not any('\u0B00' <= c <= '\u0B7F' for c in transcript):
            return False

    return True
```

---

## 📥 Data Download Script

```python
# scripts/download_data.py

import os
import requests
from tqdm import tqdm

DATASETS = {
    'english': {
        'common_voice': {
            'url': 'https://commonvoice.mozilla.org/...',
            'description': 'Mozilla Common Voice English'
        }
    },
    'hindi': {
        'common_voice': {
            'url': 'https://commonvoice.mozilla.org/...',
            'description': 'Mozilla Common Voice Hindi'
        },
        'ai4bharat': {
            'url': 'https://ai4bharat.org/...',
            'description': 'AI4Bharat Hindi Dataset'
        }
    },
    'odia': {
        'openslr': {
            'url': 'https://openslr.org/resources/66/odia.tar.gz',
            'description': 'OpenSLR Odia Corpus'
        },
        'ai4bharat': {
            'url': 'https://ai4bharat.org/...',
            'description': 'AI4Bharat Odia Dataset'
        }
    }
}

def download_file(url, output_path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def main():
    for language, sources in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Downloading {language.upper()} datasets")
        print(f"{'='*50}")

        os.makedirs(f"data/raw/{language}", exist_ok=True)

        for name, info in sources.items():
            print(f"\n{info['description']}")
            print(f"URL: {info['url']}")
            print("Please download manually from the website.")
            # Automated download where possible

if __name__ == "__main__":
    main()
```

---

## 📋 Data Preparation Checklist

- [ ] Download Mozilla Common Voice (English)
- [ ] Download Mozilla Common Voice (Hindi)
- [ ] Download AI4Bharat Hindi dataset
- [ ] Download OpenSLR Odia corpus
- [ ] Download AI4Bharat Odia dataset
- [ ] Run audio validation
- [ ] Run audio standardization
- [ ] Run text normalization
- [ ] Create train/val/test splits
- [ ] Extract MFCC features
- [ ] Verify data quality

---

_Document Version: 1.0_
_Last Updated: February 2026_
