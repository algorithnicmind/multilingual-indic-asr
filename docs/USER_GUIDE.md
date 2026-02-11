# User Guide - Multilingual Indic ASR

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.8+ installed.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Testing with Mock Data (Optional)

To verify the system works without downloading large datasets:

```bash
# Generate mock audio files
python scripts/generate_mock_data.py

# Prepare data
python scripts/prepare_data.py

# Train all models
python scripts/train_all.py

# Test inference
python -c "from src.inference import transcribe; print(transcribe('data/raw/english/clips/sample_001.wav'))"
```

### 3. Using Real Data

To build a high-accuracy system, you need real speech datasets.

1.  **Download Data**:
    Run the download helper to see links and instructions:

    ```bash
    python scripts/download_data.py
    ```

    Follow the instructions to manually download datasets for English (Common Voice), Hindi, and Odia.

2.  **Organize Data**:
    Extract the files into `data/raw/` as follows:

    ```
    data/raw/
    ├── english/
    │   ├── clips/ (contains .wav files)
    │   └── validated.tsv
    ├── hindi/
    │   ├── clips/
    │   └── validated.tsv
    └── odia/
        ├── clips/
        └── validated.tsv
    ```

3.  **Process and Train**:

    ```bash
    # 1. Prepare data (preprocessing & feature extraction)
    python scripts/prepare_data.py

    # 2. Train all models (Language ID, Acoustic, Language Models)
    python scripts/train_all.py
    ```

### 4. Running the UI

Launch the graphical user interface:

```bash
python ui/app.py
```

- Click **Browse** to select a `.wav` file.
- Click **Transcribe** to see the detected language and text.

## 🛠️ Components

- **Language ID**: Detects if audio is English, Hindi, or Odia.
- **Acoustic Model**: Converts audio features to phoneme probabilities (CNN-BiLSTM).
- **Language Model**: Predicts word sequences (N-gram).
- **Decoder**: Combines acoustic and language model scores to generate text.

## 📝 Troubleshooting

- **Missing Libraries**: Run `pip install -r requirements.txt`.
- **Training Errors**: Ensure `scripts/prepare_data.py` completes successfully before training.
- **Low Accuracy**: The default models trained on mock data will have near-zero accuracy. Train on real data for results.
