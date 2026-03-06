# System Architecture

## 📋 Overview

This document provides a comprehensive view of the Multilingual Indic ASR system architecture, including data flow, component interactions, and design decisions.

---

## 🏗️ High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                     │
│                     (Upload Audio / Record / Display Results)                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                  │
│                                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────────┐   │
│  │    Audio    │──▶│   Feature    │──▶│  Language   │──▶│   Language    │   │
│  │Preprocessing│   │  Extraction  │   │Identification│   │Specific ASR   │   │
│  └─────────────┘   └──────────────┘   └─────────────┘   └───────────────┘   │
│                                                                 │            │
│                                                                 ▼            │
│                                              ┌───────────────────────────┐   │
│                                              │         Decoder           │   │
│                                              │   (Beam Search/Viterbi)   │   │
│                                              └───────────────────────────┘   │
│                                                                 │            │
└─────────────────────────────────────────────────────────────────┼────────────┘
                                                                  │
                                                                  ▼
                                                         ┌───────────────┐
                                                         │  TRANSCRIBED  │
                                                         │     TEXT      │
                                                         └───────────────┘
```

---

## 📊 Data Flow Diagram

```
                    ┌─────────────────┐
                    │   Audio Input   │
                    │   (.wav file)   │
                    └────────┬────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │     AUDIO PREPROCESSING        │
            │                                │
            │  • Load audio file             │
            │  • Convert to mono             │
            │  • Resample to 16kHz           │
            │  • Normalize amplitude         │
            │  • Remove silence              │
            │  • Apply pre-emphasis          │
            │                                │
            └────────────────┬───────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │      FEATURE EXTRACTION        │
            │                                │
            │  • Framing (25ms, 10ms hop)    │
            │  • Windowing (Hamming)         │
            │  • FFT computation             │
            │  • Mel filterbank              │
            │  • Log compression             │
            │  • DCT (MFCC)                  │
            │  • Delta coefficients          │
            │  • Delta-delta coefficients    │
            │                                │
            │  Output: [T × 39] features     │
            └────────────────┬───────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │    LANGUAGE IDENTIFICATION     │
            │                                │
            │  • Aggregate MFCC features     │
            │  • Classify with SVM/NN        │
            │  • Output: en/hi/or            │
            │                                │
            └────────────────┬───────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │   ENGLISH   │    │    HINDI    │    │    ODIA     │
   │   PIPELINE  │    │   PIPELINE  │    │   PIPELINE  │
   │             │    │             │    │             │
   │ • Acoustic  │    │ • Acoustic  │    │ • Acoustic  │
   │   Model     │    │   Model     │    │   Model     │
   │ • Language  │    │ • Language  │    │ • Language  │
   │   Model     │    │   Model     │    │   Model     │
   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                             ▼
            ┌────────────────────────────────┐
            │           DECODER              │
            │                                │
            │  • Combine AM + LM scores      │
            │  • Beam search / Viterbi       │
            │  • Output best hypothesis      │
            │                                │
            └────────────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Text Output   │
                    │  (Unicode)     │
                    └────────────────┘
```

---

## 🧩 Component Details

### 1. Audio Preprocessing Module

**Location**: `src/preprocessing/`

**Responsibilities**:

- Audio file I/O
- Signal conditioning
- Silence removal

```python
class AudioPreprocessor:
    def __init__(self, config):
        self.sample_rate = config['sample_rate']  # 16000
        self.normalize = config['normalize']

    def process(self, audio_path):
        # 1. Load audio
        # 2. Convert to mono
        # 3. Resample
        # 4. Normalize
        # 5. Remove silence
        return processed_audio
```

**Input/Output**:

```
Input:  Raw WAV file (any sample rate, mono/stereo)
Output: NumPy array [samples] @ 16kHz, normalized
```

---

### 2. Feature Extraction Module

**Location**: `src/features/`

**Responsibilities**:

- MFCC computation
- Delta features
- Feature normalization

```python
class MFCCExtractor:
    def __init__(self, config):
        self.n_mfcc = config['n_mfcc']  # 13
        self.n_fft = config['n_fft']    # 512
        self.hop_length = config['hop_length']  # 160

    def extract(self, audio):
        # 1. Compute MFCCs
        # 2. Add deltas
        # 3. Add delta-deltas
        # 4. Normalize
        return features  # [T × 39]
```

**Output Dimensions**:

```
13 MFCCs + 13 Deltas + 13 Delta-Deltas = 39 features per frame
Frame rate: 100 frames/second (10ms hop)
```

---

### 3. Language Identification Module

**Location**: `src/language_id/`

**Architecture Options**:

#### Option A: SVM Classifier

```
┌─────────────┐
│   MFCCs     │
│  [T × 39]   │
└──────┬──────┘
       │ Temporal Aggregation
       ▼
┌─────────────┐
│   Mean +    │
│   Variance  │
│  [1 × 78]   │
└──────┬──────┘
       │ SVM Classification
       ▼
┌─────────────┐
│  Language   │
│  en/hi/or   │
└─────────────┘
```

#### Option B: Neural Network

```
┌─────────────┐
│   MFCCs     │
│  [T × 39]   │
└──────┬──────┘
       │ LSTM Encoding
       ▼
┌─────────────┐
│   Hidden    │
│   [1 × 128] │
└──────┬──────┘
       │ Dense + Softmax
       ▼
┌─────────────┐
│  Language   │
│  Probs [3]  │
└─────────────┘
```

---

### 4. Acoustic Model

**Location**: `src/acoustic_model/`

**Architecture**: CNN + BiLSTM + CTC

```
┌─────────────────────────────────────────────────────────┐
│                    ACOUSTIC MODEL                        │
│                                                         │
│   Input: MFCC Features [Batch × Time × 39]              │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              CNN Feature Extractor               │   │
│   │                                                 │   │
│   │   Conv2D(1→32, 3×3) → ReLU → BatchNorm         │   │
│   │   Conv2D(32→64, 3×3) → ReLU → BatchNorm        │   │
│   │   Conv2D(64→128, 3×3) → ReLU → BatchNorm       │   │
│   │   MaxPool → Dropout(0.3)                        │   │
│   │                                                 │   │
│   └─────────────────────┬───────────────────────────┘   │
│                         │                               │
│   ┌─────────────────────▼───────────────────────────┐   │
│   │              BiLSTM Sequence Encoder             │   │
│   │                                                 │   │
│   │   BiLSTM(256, 3 layers) → Dropout(0.3)         │   │
│   │                                                 │   │
│   └─────────────────────┬───────────────────────────┘   │
│                         │                               │
│   ┌─────────────────────▼───────────────────────────┐   │
│   │              Output Layer                        │   │
│   │                                                 │   │
│   │   Linear(512 → num_phonemes + 1)               │   │
│   │   LogSoftmax                                    │   │
│   │                                                 │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│   Output: Log Probabilities [Batch × Time × Vocab]      │
│                                                         │
│   Loss: CTC Loss (Connectionist Temporal Classification)│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**One model per language** to handle language-specific phoneme inventories.

---

### 5. Language Model

**Location**: `src/language_model/`

**Architecture**: N-gram with Kneser-Ney Smoothing

```
┌─────────────────────────────────────────────────────────┐
│                    LANGUAGE MODEL                        │
│                                                         │
│   Type: Trigram (N=3)                                   │
│                                                         │
│   P(w₃ | w₁, w₂) = Interpolated Kneser-Ney              │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │   Storage Structure                              │   │
│   │                                                 │   │
│   │   unigrams: { word: count }                     │   │
│   │   bigrams:  { (w1, w2): count }                 │   │
│   │   trigrams: { (w1, w2, w3): count }             │   │
│   │                                                 │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│   Vocabulary: Top 50,000 words per language             │
│   OOV Handling: <UNK> token                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 6. Decoder Module

**Location**: `src/decoder/`

**Algorithm**: Beam Search with LM Integration

```
┌─────────────────────────────────────────────────────────┐
│                      DECODER                             │
│                                                         │
│   Input:                                                │
│   - Acoustic Model Output: P(y|x) [T × Vocab]           │
│   - Language Model: P(w)                                │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │            Beam Search Algorithm                 │   │
│   │                                                 │   │
│   │   beam_width = 10                               │   │
│   │                                                 │   │
│   │   For each time step t:                         │   │
│   │     For each beam hypothesis h:                 │   │
│   │       For each vocabulary token v:              │   │
│   │         score = α * log P(v|x) + β * log P(v|h)│   │
│   │         Add (h + v, score) to candidates        │   │
│   │     Keep top beam_width candidates              │   │
│   │                                                 │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│   α = AM weight (0.6)                                   │
│   β = LM weight (0.4)                                   │
│                                                         │
│   Output: Best text hypothesis                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
│                                                                             │
│   ┌─────────────────┐                                                       │
│   │  Raw Datasets   │                                                       │
│   │  (Audio + Text) │                                                       │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐   │
│   │ Data Preparation │──────▶│ Language ID     │──────▶│ Acoustic Model  │   │
│   │                 │       │ Training        │      │ Training        │   │
│   │ • Split by lang │       │                 │      │                 │   │
│   │ • Preprocess    │       │ • Extract MFCCs │      │ • CTC Loss      │   │
│   │ • Extract feats │       │ • Train SVM/NN  │      │ • Backprop      │   │
│   │ • Create splits │       │ • Validate      │      │ • Validate      │   │
│   └─────────────────┘       └─────────────────┘      └─────────────────┘   │
│                                                                │            │
│   ┌─────────────────┐                                          │            │
│   │ Text Corpora    │                                          │            │
│   │ (per language)  │                                          │            │
│   └────────┬────────┘                                          │            │
│            │                                                   │            │
│            ▼                                                   ▼            │
│   ┌─────────────────┐                                 ┌─────────────────┐   │
│   │ Language Model  │                                 │  Model Saving   │   │
│   │ Training        │                                 │                 │   │
│   │                 │                                 │ • Checkpoints   │   │
│   │ • Tokenize      │────────────────────────────────▶│ • Best models   │   │
│   │ • Count N-grams │                                 │ • Configs       │   │
│   │ • Apply smooth. │                                 │                 │   │
│   └─────────────────┘                                 └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
multilingual-indic-asr/
│
├── src/
│   ├── __init__.py
│   │
│   ├── preprocessing/           # Audio preprocessing
│   │   ├── __init__.py
│   │   ├── audio_utils.py      # Audio I/O utilities
│   │   └── preprocessor.py     # Main preprocessing class
│   │
│   ├── features/                # Feature extraction
│   │   ├── __init__.py
│   │   └── mfcc_extractor.py   # MFCC extraction class
│   │
│   ├── language_id/             # Language identification
│   │   ├── __init__.py
│   │   ├── model.py            # LID model definition
│   │   └── train.py            # Training script
│   │
│   ├── acoustic_model/          # Acoustic modeling
│   │   ├── __init__.py
│   │   ├── model.py            # CNN-BiLSTM model
│   │   ├── train.py            # Training script
│   │   └── dataset.py          # Data loading
│   │
│   ├── language_model/          # Language modeling
│   │   ├── __init__.py
│   │   └── ngram.py            # N-gram LM
│   │
│   ├── decoder/                 # Decoding
│   │   ├── __init__.py
│   │   ├── viterbi.py          # Viterbi decoder
│   │   └── beam_search.py      # Beam search decoder
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py          # WER, CER calculation
│   │   └── visualization.py    # Plotting functions
│   │
│   └── inference.py             # End-to-end inference
│
├── models/                      # Saved models
│   ├── language_id/
│   ├── acoustic/
│   │   ├── english/
│   │   ├── hindi/
│   │   └── odia/
│   └── language/
│
├── data/                        # Data storage
├── ui/                          # User interface
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── scripts/                     # Utility scripts
│   ├── download_data.py
│   ├── organize_dataset.py     # Dataset organization
│   ├── fix_tsv.py              # TSV formatting fixes
│   ├── prepare_data.py
│   └── train_all.py            # Unified training script
└── docs/                        # Documentation
```

---

## 🔌 API Design

### Main Pipeline Class

```python
class ASRPipeline:
    """
    End-to-end ASR pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.preprocessor = AudioPreprocessor(self.config)
        self.feature_extractor = MFCCExtractor(self.config)
        self.language_identifier = LanguageIdentifier(self.config)
        self.acoustic_models = {
            'english': AcousticModel.load('english'),
            'hindi': AcousticModel.load('hindi'),
            'odia': AcousticModel.load('odia')
        }
        self.language_models = {
            'english': LanguageModel.load('english'),
            'hindi': LanguageModel.load('hindi'),
            'odia': LanguageModel.load('odia')
        }
        self.decoder = BeamSearchDecoder(self.config)

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to WAV file

        Returns:
            {
                'language': 'en' | 'hi' | 'or',
                'text': str,
                'confidence': float
            }
        """
        # 1. Preprocess audio
        audio = self.preprocessor.process(audio_path)

        # 2. Extract features
        features = self.feature_extractor.extract(audio)

        # 3. Identify language
        language = self.language_identifier.predict(features)

        # 4. Get acoustic model output
        am_output = self.acoustic_models[language].forward(features)

        # 5. Decode with language model
        text = self.decoder.decode(
            am_output,
            self.language_models[language]
        )

        return {
            'language': language,
            'text': text,
            'confidence': self.decoder.get_confidence()
        }
```

---

## 🔄 Inter-Component Communication

| Source            | Target            | Data Format           |
| ----------------- | ----------------- | --------------------- |
| Audio Input       | Preprocessor      | WAV file path         |
| Preprocessor      | Feature Extractor | NumPy array [samples] |
| Feature Extractor | Language ID       | NumPy array [T × 39]  |
| Feature Extractor | Acoustic Model    | Tensor [B × T × 39]   |
| Language ID       | Router            | Language string       |
| Acoustic Model    | Decoder           | Tensor [B × T × V]    |
| Language Model    | Decoder           | Probability function  |
| Decoder           | Output            | Text string           |

---

## ⚡ Performance Considerations

### Memory Optimization

- Streaming feature extraction for long audio
- Batch processing for inference
- Model quantization (future)

### Speed Optimization

- GPU acceleration for neural networks
- Efficient N-gram lookup with tries
- Beam pruning for decoder

### Scalability

- Modular design for easy extension
- Plugin architecture for new languages
- Configuration-driven behavior

---

_Document Version: 1.0_
_Last Updated: February 2026_
