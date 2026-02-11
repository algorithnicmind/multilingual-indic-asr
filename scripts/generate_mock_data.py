#!/usr/bin/env python
"""
Generate mock data to test the ASR pipeline.

This script creates:
1. Directory structure for data
2. Dummy .wav files (sine waves)
3. Dummy transcripts (validated.tsv)
"""

import os
import numpy as np
import soundfile as sf
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

LANGUAGES = ['english', 'hindi', 'odia']
NUM_SAMPLES = 10
SAMPLE_RATE = 16000

def generate_sine_wave(duration, frequency, sample_rate=16000, amplitude=0.5):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return amplitude * np.sin(2 * np.pi * frequency * t)

def create_mock_data():
    """Create mock audio files and transcripts."""
    logger.info("Generating mock data...")
    
    base_dir = Path('data/raw')
    
    for language in LANGUAGES:
        logger.info(f"Processing {language}...")
        
        # Create directories
        clips_dir = base_dir / language / 'clips'
        clips_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcript list
        transcripts = []
        
        for i in range(1, NUM_SAMPLES + 1):
            filename = f"sample_{i:03d}.wav"
            file_path = clips_dir / filename
            
            # Generate random audio
            duration = random.uniform(1.0, 3.0)  # 1 to 3 seconds
            freq = random.uniform(200, 800)      # 200Hz to 800Hz
            audio = generate_sine_wave(duration, freq, SAMPLE_RATE)
            
            # Save audio
            sf.write(file_path, audio, SAMPLE_RATE)
            
            # Generate dummy transcript
            if language == 'english':
                text = f"this is sample number {i}"
            elif language == 'hindi':
                 text = f"यह नमूना नंबर {i} है" # This is sample number i
            elif language == 'odia':
                 text = f"ଏହା ହେଉଛି ନମୁନା ନମ୍ବର {i}" # This is sample number i (approx)
            
            transcripts.append(f"client_{i}\tclips/{filename}\t{text}\t0\t0\t20\tmale\tstd")
            
        # Save validated.tsv
        tsv_path = base_dir / language / 'validated.tsv'
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write("client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\n")
            f.write('\n'.join(transcripts))
            
        logger.info(f"Created {NUM_SAMPLES} samples for {language}")

    logger.info("Mock data generation complete!")

if __name__ == "__main__":
    create_mock_data()
