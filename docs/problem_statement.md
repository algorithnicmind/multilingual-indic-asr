# Problem Statement

## 📋 Overview

Speech recognition technology has become ubiquitous in modern applications, from virtual assistants to transcription services. However, most available solutions rely heavily on:

- Proprietary cloud APIs (Google Speech, Azure, AWS)
- Large pretrained models (Whisper, DeepSpeech)
- English-centric training data

This project addresses the need for a **transparent, educational, and locally-deployable** speech recognition system for **Indian languages**.

---

## 🎯 Problem Definition

### The Challenge

Build a **multilingual Automatic Speech Recognition (ASR)** system from scratch that:

1. **Converts spoken audio** in English, Hindi, and Odia to text
2. **Automatically identifies** the spoken language
3. **Operates entirely locally** without external API dependencies
4. **Uses no pretrained models** - all training done from raw data

### Input Specifications

| Parameter       | Requirement      |
| --------------- | ---------------- |
| **Format**      | WAV audio files  |
| **Sample Rate** | 16 kHz           |
| **Channels**    | Mono             |
| **Duration**    | 0.5 - 30 seconds |

### Output Specifications

| Output                | Format                                      |
| --------------------- | ------------------------------------------- |
| **Detected Language** | `en` (English) / `hi` (Hindi) / `or` (Odia) |
| **Transcription**     | Unicode text string                         |
| **Confidence Score**  | Float (0.0 - 1.0)                           |

---

## 🔒 Core Constraints

### ❌ What We Cannot Use

| Constraint                | Rationale                |
| ------------------------- | ------------------------ |
| **Google Speech API**     | External dependency      |
| **OpenAI Whisper**        | Pretrained model         |
| **Azure Speech Services** | Cloud-based API          |
| **AWS Transcribe**        | Cloud-based API          |
| **DeepSpeech**            | Pretrained model         |
| **Wav2Vec**               | Pretrained model         |
| **Any pretrained ASR**    | Defeats learning purpose |

### ✅ What We Can Use

| Allowed          | Purpose                       |
| ---------------- | ----------------------------- |
| **NumPy, SciPy** | Numerical computing           |
| **Librosa**      | Audio signal processing       |
| **Scikit-learn** | ML algorithms                 |
| **PyTorch**      | Neural network implementation |
| **Raw datasets** | Training data                 |

> 💡 **Key Principle**: Libraries are tools for computation, not intelligence. We use them to implement algorithms, not to import pre-trained solutions.

---

## 🌐 Scope

### In Scope

1. **Audio File Processing**
   - WAV file input
   - Preprocessing and normalization
   - Feature extraction

2. **Language Identification**
   - Automatic detection of English, Hindi, or Odia
   - Language-specific model routing

3. **Speech-to-Text Conversion**
   - Phoneme recognition
   - Word sequence prediction
   - Text output generation

4. **User Interface**
   - File upload capability
   - Language display
   - Transcription output

### Out of Scope (Initial Version)

1. **Real-time streaming** (future enhancement)
2. **Speaker diarization** (who said what)
3. **Punctuation and formatting**
4. **Additional languages** (beyond English, Hindi, Odia)
5. **Noise-robust recognition** (assumes clean audio)

---

## 🎓 Educational Objectives

This project is designed as a **learning exercise** to understand:

### Signal Processing

- Digital audio fundamentals
- Time-domain vs frequency-domain analysis
- Filter banks and spectrograms

### Feature Engineering

- Mel-frequency cepstral coefficients (MFCC)
- Delta and delta-delta features
- Feature normalization techniques

### Machine Learning

- Classification algorithms (SVM, Neural Networks)
- Sequence modeling (HMM, LSTM)
- Training and validation strategies

### Speech Recognition Theory

- Acoustic modeling
- Language modeling
- Decoding algorithms

---

## 📊 Success Criteria

### Minimum Viable Product (MVP)

| Metric                      | Target              |
| --------------------------- | ------------------- |
| Language Detection Accuracy | ≥ 80%               |
| Word Error Rate (English)   | ≤ 50%               |
| Word Error Rate (Hindi)     | ≤ 55%               |
| Word Error Rate (Odia)      | ≤ 60%               |
| Processing Speed            | ≤ 2x audio duration |

### Stretch Goals

| Metric                          | Target              |
| ------------------------------- | ------------------- |
| Language Detection Accuracy     | ≥ 90%               |
| Word Error Rate (All Languages) | ≤ 35%               |
| Processing Speed                | ≤ 1x audio duration |

---

## ⚠️ Realistic Expectations

### This is NOT

- A production-ready system
- A competitor to commercial ASR
- Perfect or error-free

### This IS

- An educational project
- A demonstration of ASR fundamentals
- A starting point for further learning
- Acceptable at 60-70% accuracy

---

## 📚 References

1. Jurafsky, D., & Martin, J. H. - "Speech and Language Processing"
2. Rabiner, L. R. - "A Tutorial on Hidden Markov Models"
3. Graves, A. - "Connectionist Temporal Classification"
4. Mozilla Common Voice Documentation
5. AI4Bharat Research Papers

---

_Document Version: 1.0_
_Last Updated: February 2026_
