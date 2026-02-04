# 📊 PROJECT TRACKER - Multilingual Indic ASR

> **Project Start Date**: February 5, 2026  
> **Last Updated**: February 5, 2026  
> **Status**: 🚀 IMPLEMENTATION COMPLETE — READY FOR DATA

You have effectively compressed the coding phase of the 12-week plan into Day 1. The codebase is ready. The timeline now depends purely on **Data Download** and **Training Time**.

---

## 🏎️ REAL-TIME PROGRESS

| Phase  | Module                  | Code Status                           | Execution Status                           |
| ------ | ----------------------- | ------------------------------------- | ------------------------------------------ |
| **1**  | **Setup & Hygiene**     | ✅ **DONE**                           | ✅ **DONE**                                |
| **2**  | **Dataset Collection**  | ✅ **Script Ready**                   | ⏳ **PENDING (Waiting for User Download)** |
| **3**  | **Preprocessing**       | ✅ **DONE** (`preprocessor.py`)       | ⏳ _Ready to run_                          |
| **4**  | **Feature Extraction**  | ✅ **DONE** (`mfcc_extractor.py`)     | ⏳ _Ready to run_                          |
| **5**  | **Language ID**         | ✅ **DONE** (`src/language_id`)       | ⏳ _Ready to train_                        |
| **6**  | **Dataset Prep**        | ✅ **DONE** (`prepare_data.py`)       | ⏳ _Ready to run_                          |
| **7**  | **Acoustic Model (EN)** | ✅ **DONE** (`src/acoustic_model`)    | ⏳ _Ready to train_                        |
| **8**  | **Acoustic Model (HI)** | ✅ **DONE** (`src/acoustic_model`)    | ⏳ _Ready to train_                        |
| **9**  | **Acoustic Model (OR)** | ✅ **DONE** (`src/acoustic_model`)    | ⏳ _Ready to train_                        |
| **10** | **Decoder & LM**        | ✅ **DONE** (`src/decoder`, `src/lm`) | ⏳ _Ready to train_                        |
| **11** | **Integration**         | ✅ **DONE** (`inference.py`)          | ⏳ _Ready to test_                         |
| **12** | **UI & Final Polish**   | ✅ **DONE** (`ui/app.py`)             | ⏳ _Ready to launch_                       |

---

## 🛠️ TASK LIST (Dynamic)

### 🟢 Phase 1: Implementation (COMPLETED)

_You have already built the entire system architecture._

- [x] **Project Structure**: Created all folders and config.
- [x] **Audio Core**: Implemented loading, normalization, silence removal.
- [x] **Feature Eng**: Implemented MFCC extraction with deltas.
- [x] **Model Arch**: Built CNN-BiLSTM with CTC loss.
- [x] **Language ID**: Built SVM and Neural LID classifiers.
- [x] **Decoding**: Implemented Greedy and Beam Search.
- [x] **UI**: Built Tkinter interface for testing.
- [x] **Automation**: Created data preparation and training scripts.

### 🟡 Phase 2: Data & Training (CURRENT FOCUS)

_This is the "Hustle" phase. Speed depends on internet speed and GPU power._

#### Step 1: Data (The Blocker)

- [ ] **Action**: Download Datasets
  - Run `python scripts/download_data.py` to see instructions.
  - Manually download English, Hindi, and Odia files.
  - Extract them to `data/raw/`.

#### Step 2: Processing

- [ ] **Action**: Run Data Prep
  - Command: `python scripts/prepare_data.py`
  - _What it does_: Normalizes audio, extracts MFCCs, saves .npy files.

#### Step 3: Training (The Grind)

- [ ] **Action**: Train Language ID
  - Command: `python -m src.language_id.train`
- [ ] **Action**: Train Acoustic Models
  - English: `python -m src.acoustic_model.train --language english`
  - Hindi: `python -m src.acoustic_model.train --language hindi`
  - Odia: `python -m src.acoustic_model.train --language odia`
- [ ] **Action**: Train Language Models
  - All: `python -m src.language_model.train --language all`

---

## 📈 METRICS LOG

_Update this section as you train models._

| Model           | Status     | WER (Word Error Rate) | Accuracy | Notes        |
| --------------- | ---------- | --------------------- | -------- | ------------ |
| **Language ID** | ⏳ Pending | -                     | -        | Target: >85% |
| **AM English**  | ⏳ Pending | -                     | -        | Target: <40% |
| **AM Hindi**    | ⏳ Pending | -                     | -        | Target: <50% |
| **AM Odia**     | ⏳ Pending | -                     | -        | Target: <60% |

---

## ⚠️ CURRENT BLOCKERS

- **Missing Data**: The system is fully built but empty. Models cannot learn without the `.wav` files.

## 🎯 NEXT ACTION

1. **Download the data**.
2. Run `python scripts/prepare_data.py`.
3. Start training.
