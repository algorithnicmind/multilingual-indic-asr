#!/usr/bin/env python
"""
Prepare data for training.

This script:
1. Validates audio files
2. Preprocesses and standardizes audio
3. Extracts MFCC features
4. Creates train/val/test splits
"""

import os
import sys
from pathlib import Path
import numpy as np
import random
import logging
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import AudioPreprocessor, validate_audio, save_audio
from src.features import MFCCExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


LANGUAGES = ['english', 'hindi', 'odia']
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}


def load_transcript(transcript_path: Path) -> dict:
    """Load transcripts from TSV file."""
    transcripts = {}
    
    if transcript_path.suffix == '.tsv':
        with open(transcript_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            path_idx = header.index('path') if 'path' in header else 1
            sent_idx = header.index('sentence') if 'sentence' in header else 2
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > max(path_idx, sent_idx):
                    audio_file = parts[path_idx]
                    sentence = parts[sent_idx]
                    transcripts[audio_file] = sentence
    
    return transcripts


def process_language(language: str, config: dict):
    """Process all data for a language."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing {language.upper()}")
    logger.info(f"{'='*50}")
    
    raw_dir = Path('data/raw') / language
    processed_dir = Path('data/processed') / language
    features_dir = Path('data/features') / language
    
    # Check if raw data exists
    clips_dir = raw_dir / 'clips'
    if not clips_dir.exists():
        logger.warning(f"No clips found for {language}. Skipping...")
        return
    
    # Load transcripts
    transcript_path = raw_dir / 'validated.tsv'
    if transcript_path.exists():
        transcripts = load_transcript(transcript_path)
        logger.info(f"Loaded {len(transcripts)} transcripts")
    else:
        transcripts = {}
        logger.warning("No transcript file found")
    
    # Initialize processors
    preprocessor = AudioPreprocessor(config)
    feature_extractor = MFCCExtractor(config)
    
    # Get audio files
    audio_files = list(clips_dir.glob('*.wav'))
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        return
    
    # Validate and process files
    valid_samples = []
    
    for audio_path in tqdm(audio_files, desc="Validating"):
        is_valid, message = validate_audio(str(audio_path))
        if is_valid:
            valid_samples.append(audio_path)
    
    logger.info(f"Valid samples: {len(valid_samples)}/{len(audio_files)}")
    
    # Create splits
    random.shuffle(valid_samples)
    n = len(valid_samples)
    
    train_end = int(n * SPLIT_RATIOS['train'])
    val_end = train_end + int(n * SPLIT_RATIOS['val'])
    
    splits = {
        'train': valid_samples[:train_end],
        'val': valid_samples[train_end:val_end],
        'test': valid_samples[val_end:]
    }
    
    for split_name, samples in splits.items():
        logger.info(f"{split_name}: {len(samples)} samples")
    
    # Process each split
    for split_name, samples in splits.items():
        split_processed_dir = processed_dir / split_name
        split_features_dir = features_dir / split_name
        split_processed_dir.mkdir(parents=True, exist_ok=True)
        split_features_dir.mkdir(parents=True, exist_ok=True)
        
        split_transcripts = []
        
        for audio_path in tqdm(samples, desc=f"Processing {split_name}"):
            try:
                # Get filename
                filename = audio_path.stem
                
                # Preprocess audio
                audio = preprocessor.process(str(audio_path))
                
                # Save processed audio
                processed_path = split_processed_dir / f"{filename}.wav"
                save_audio(audio, str(processed_path), 16000)
                
                # Extract features
                features = feature_extractor.extract(audio)
                
                # Save features
                features_path = split_features_dir / f"{filename}.npy"
                np.save(features_path, features)
                
                # Get transcript
                clip_name = f"clips/{audio_path.name}"
                transcript = transcripts.get(clip_name, "")
                split_transcripts.append(f"{filename}\t{transcript}")
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
        
        # Save transcripts
        transcript_file = Path('data/transcripts') / language / f"{split_name}.txt"
        transcript_file.parent.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(split_transcripts))
        
        logger.info(f"Saved {len(split_transcripts)} transcripts to {transcript_file}")


def main():
    """Main entry point."""
    print("\n🎙️ Multilingual Indic ASR - Data Preparation")
    print("="*50)
    
    # Configuration
    config = {
        'sample_rate': 16000,
        'normalize': True,
        'remove_silence': True,
        'n_mfcc': 13,
        'use_delta': True,
        'use_delta_delta': True,
        'normalize_features': True
    }
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process each language
    for language in LANGUAGES:
        process_language(language, config)
    
    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
