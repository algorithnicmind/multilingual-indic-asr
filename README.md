# 🎙️ Multilingual Indic ASR System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**A from-scratch multilingual speech-to-text system for English, Hindi, and Odia**

[Overview](#-overview) • [Features](#-features) • [Architecture](#-system-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

This project implements a **complete Automatic Speech Recognition (ASR) pipeline** from scratch, designed to convert spoken audio in **English**, **Hindi**, and **Odia** into text.

### 🎯 Key Highlights

- **No Third-Party Speech APIs**: No Google Speech, Whisper, Azure, AWS, or any external ASR services
- **No Pretrained Models**: All models are trained from scratch using raw datasets
- **Complete Pipeline**: From audio preprocessing to final transcription
- **Multilingual Support**: Handles three languages with automatic language detection
- **Educational Purpose**: Designed to understand the inner workings of ASR systems

---

## 🌟 Features

| Feature                    | Description                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| 🔊 **Audio Preprocessing** | Mono conversion, 16kHz resampling, normalization, silence removal |
| 📊 **Feature Extraction**  | MFCC with delta and delta-delta coefficients                      |
| 🌐 **Language Detection**  | Automatic identification of English, Hindi, or Odia               |
| 🧠 **Acoustic Modeling**   | Separate CNN+BiLSTM models per language                           |
| 📝 **Language Modeling**   | N-gram based language models for each language                    |
| 🔤 **Decoding**            | Viterbi/Beam search for optimal transcription                     |
| 🖥️ **User Interface**      | Simple UI for audio upload and transcription display              |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                              │
│                    (.wav, 16kHz, Mono)                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                              │
│         • Mono Conversion  • Resampling  • Normalization        │
│                     • Silence Removal                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                           │
│              MFCC (13-40 coefficients)                          │
│              + Delta + Delta-Delta                              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGUAGE IDENTIFICATION                        │
│                   SVM / Neural Network                          │
│                                                                 │
│              ┌─────────┬─────────┬─────────┐                    │
│              │ English │  Hindi  │   Odia  │                    │
│              └─────────┴─────────┴─────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  English ASR    │ │   Hindi ASR     │ │   Odia ASR      │
│                 │ │                 │ │                 │
│ Acoustic Model  │ │ Acoustic Model  │ │ Acoustic Model  │
│ Language Model  │ │ Language Model  │ │ Language Model  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER                                  │
│              Viterbi / Beam Search Algorithm                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TEXT OUTPUT                               │
│              Transcribed text in detected language              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

### Core Language

- **Python 3.10+**

### Libraries & Frameworks

| Category                | Libraries                     |
| ----------------------- | ----------------------------- |
| **Numerical Computing** | NumPy, SciPy                  |
| **Audio Processing**    | Librosa, SoundDevice, PyAudio |
| **Machine Learning**    | Scikit-learn                  |
| **Deep Learning**       | PyTorch                       |
| **Visualization**       | Matplotlib                    |
| **User Interface**      | Tkinter                       |

> ⚠️ **Note**: These libraries are tools for implementation, not intelligence. All model architectures and training are done from scratch.

---

## 📚 Datasets

### English

| Dataset              | Source                                                     |
| -------------------- | ---------------------------------------------------------- |
| Mozilla Common Voice | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) |

### Hindi

| Dataset                      | Source                                                     |
| ---------------------------- | ---------------------------------------------------------- |
| Mozilla Common Voice (Hindi) | [commonvoice.mozilla.org](https://commonvoice.mozilla.org) |
| AI4Bharat Hindi              | [ai4bharat.org](https://ai4bharat.org)                     |

### Odia

| Dataset               | Source                                 |
| --------------------- | -------------------------------------- |
| AI4Bharat Odia        | [ai4bharat.org](https://ai4bharat.org) |
| OpenSLR Indic Corpora | [openslr.org](https://openslr.org)     |

### Data Requirements

Each dataset entry must contain:

- ✅ Audio file (.wav format)
- ✅ Exact text transcript

---

## 📁 Project Structure

```
multilingual-indic-asr/
│
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
├── 📄 config.yaml                  # Configuration file
│
├── 📂 data/                        # Dataset storage
│   ├── raw/                        # Raw audio files
│   │   ├── english/
│   │   ├── hindi/
│   │   └── odia/
│   ├── processed/                  # Preprocessed data
│   │   ├── english/
│   │   ├── hindi/
│   │   └── odia/
│   └── transcripts/                # Text transcripts
│       ├── english/
│       ├── hindi/
│       └── odia/
│
├── 📂 src/                         # Source code
│   ├── __init__.py
│   ├── preprocessing/              # Audio preprocessing
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   └── preprocessor.py
│   ├── features/                   # Feature extraction
│   │   ├── __init__.py
│   │   └── mfcc_extractor.py
│   ├── language_id/                # Language identification
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── train.py
│   ├── acoustic_model/             # Acoustic models
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── models/
│   │       ├── english/
│   │       ├── hindi/
│   │       └── odia/
│   ├── language_model/             # Language models
│   │   ├── __init__.py
│   │   ├── ngram.py
│   │   └── models/
│   │       ├── english/
│   │       ├── hindi/
│   │       └── odia/
│   ├── decoder/                    # Decoding algorithms
│   │   ├── __init__.py
│   │   ├── viterbi.py
│   │   └── beam_search.py
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
│
├── 📂 models/                      # Saved model weights
│   ├── language_id/
│   ├── acoustic/
│   │   ├── english/
│   │   ├── hindi/
│   │   └── odia/
│   └── language/
│       ├── english/
│       ├── hindi/
│       └── odia/
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── 📂 tests/                       # Unit tests
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── 📂 ui/                          # User interface
│   ├── __init__.py
│   └── app.py
│
├── 📂 scripts/                     # Utility scripts
│   ├── download_data.py
│   ├── prepare_data.py
│   └── train_all.py
│
└── 📂 docs/                        # Documentation
    ├── problem_statement.md
    ├── system_architecture.md
    ├── dataset_details.md
    ├── feature_extraction.md
    ├── language_identification.md
    ├── acoustic_model.md
    ├── language_model.md
    ├── decoder.md
    ├── evaluation.md
    └── future_work.md
```

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/multilingual-indic-asr.git
cd multilingual-indic-asr
```

2. **Create virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download datasets**

```bash
python scripts/download_data.py
```

5. **Prepare data**

```bash
python scripts/prepare_data.py
```

---

## 🚀 Usage

### Training

#### Train Language Identification Model

```bash
python -m src.language_id.train
```

#### Train Acoustic Models

```bash
# Train all languages
python -m src.acoustic_model.train --language all

# Train specific language
python -m src.acoustic_model.train --language english
python -m src.acoustic_model.train --language hindi
python -m src.acoustic_model.train --language odia
```

#### Train Language Models

```bash
python -m src.language_model.ngram --language all
```

### Inference

#### Command Line

```bash
python -m src.inference --audio path/to/audio.wav
```

#### Using the UI

```bash
python -m ui.app
```

### Example Code

```python
from src.preprocessing import AudioPreprocessor
from src.features import MFCCExtractor
from src.language_id import LanguageIdentifier
from src.inference import ASRPipeline

# Initialize pipeline
pipeline = ASRPipeline()

# Transcribe audio
result = pipeline.transcribe("audio.wav")

print(f"Detected Language: {result['language']}")
print(f"Transcription: {result['text']}")
```

---

## 📊 Evaluation Metrics

| Metric                          | Description                     | Target            |
| ------------------------------- | ------------------------------- | ----------------- |
| **Word Error Rate (WER)**       | Measures transcription accuracy | 30-40%            |
| **Language Detection Accuracy** | Correct language identification | 85%+              |
| **Inference Latency**           | Time to process audio           | < 2x audio length |

> 📝 **Note**: Target accuracy of 60-70% is acceptable for a student-built from-scratch ASR system.

---

## 📖 Documentation

Detailed documentation is available in the `/docs` folder:

| Document                                                   | Description                                |
| ---------------------------------------------------------- | ------------------------------------------ |
| [Problem Statement](docs/problem_statement.md)             | Problem definition, scope, and constraints |
| [System Architecture](docs/system_architecture.md)         | Detailed pipeline and data flow            |
| [Dataset Details](docs/dataset_details.md)                 | Dataset sources and preprocessing          |
| [Feature Extraction](docs/feature_extraction.md)           | MFCC and feature engineering               |
| [Language Identification](docs/language_identification.md) | Language detection model                   |
| [Acoustic Model](docs/acoustic_model.md)                   | Speech-to-phoneme modeling                 |
| [Language Model](docs/language_model.md)                   | N-gram language modeling                   |
| [Decoder](docs/decoder.md)                                 | Viterbi and beam search                    |
| [Evaluation](docs/evaluation.md)                           | Metrics and results                        |
| [Future Work](docs/future_work.md)                         | Planned improvements                       |

---

## 🎯 Motivation

### Why Build from Scratch?

1. **Deep Understanding**: Learn the fundamental principles of speech recognition
2. **No Black Boxes**: Understand every component of the ASR pipeline
3. **Educational Value**: Perfect for academic projects and learning
4. **Customization**: Full control over every aspect of the system
5. **Indic Language Focus**: Specialized for Indian languages

### Learning Objectives

- Digital signal processing for audio
- Feature extraction techniques (MFCC)
- Machine learning for classification
- Sequential modeling with RNNs/LSTMs
- Language modeling with N-grams
- Decoding algorithms (Viterbi, Beam Search)

---

## 📈 Results & Limitations

### Current Results (Expected)

| Language | WER  | Language Detection |
| -------- | ---- | ------------------ |
| English  | ~35% | 90%                |
| Hindi    | ~45% | 85%                |
| Odia     | ~50% | 80%                |

### Known Limitations

- ⚠️ Not production-ready (educational project)
- ⚠️ Lower accuracy than commercial systems
- ⚠️ Limited vocabulary coverage
- ⚠️ Sensitive to background noise
- ⚠️ Requires significant training time

---

## 🔮 Future Roadmap

- [x] **Phase 1**: Core pipeline implementation
- [x] **Phase 2**: Basic model training
- [x] **Phase 3**: UI development
- [ ] **Phase 4**: Optimization and evaluation
- [x] **Phase 5**: Real-time microphone support
- [ ] **Phase 6**: Additional language support

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Mozilla Common Voice for English and Hindi datasets
- AI4Bharat for Indian language resources
- OpenSLR for Indic speech corpora
- The open-source community

---

<div align="center">

**Made with ❤️ for Indic Languages**

[⬆ Back to Top](#️-multilingual-indic-asr-system)

</div>
