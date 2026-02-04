#!/usr/bin/env python
"""
Train all models for the ASR system.

This script:
1. Trains the language identification model
2. Trains acoustic models for each language
3. Trains language models for each language
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_language_id():
    """Train language identification model."""
    logger.info("\n" + "="*50)
    logger.info("Training Language Identification Model")
    logger.info("="*50)
    
    from src.language_id.train import train_lid_model
    train_lid_model()


def train_acoustic_models(languages=None):
    """Train acoustic models for specified languages."""
    logger.info("\n" + "="*50)
    logger.info("Training Acoustic Models")
    logger.info("="*50)
    
    from src.acoustic_model.train import train_acoustic_model
    
    all_languages = ['english', 'hindi', 'odia']
    languages = languages or all_languages
    
    for lang in languages:
        logger.info(f"\nTraining {lang} acoustic model...")
        train_acoustic_model(lang)


def train_language_models(languages=None):
    """Train language models for specified languages."""
    logger.info("\n" + "="*50)
    logger.info("Training Language Models")
    logger.info("="*50)
    
    from src.language_model.train import train_language_model
    
    all_languages = ['english', 'hindi', 'odia']
    languages = languages or all_languages
    
    for lang in languages:
        logger.info(f"\nTraining {lang} language model...")
        train_language_model(lang)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ASR models')
    parser.add_argument('--component', type=str, choices=['lid', 'am', 'lm', 'all'],
                       default='all', help='Component to train')
    parser.add_argument('--language', type=str, nargs='+',
                       choices=['english', 'hindi', 'odia', 'all'],
                       default=['all'], help='Languages to train')
    
    args = parser.parse_args()
    
    print("\n🎙️ Multilingual Indic ASR - Training Pipeline")
    print("="*50)
    
    # Resolve languages
    if 'all' in args.language:
        languages = ['english', 'hindi', 'odia']
    else:
        languages = args.language
    
    # Train components
    if args.component in ['lid', 'all']:
        try:
            train_language_id()
        except ImportError:
            logger.warning("Language ID training module not available")
    
    if args.component in ['am', 'all']:
        try:
            train_acoustic_models(languages)
        except ImportError:
            logger.warning("Acoustic model training module not available")
    
    if args.component in ['lm', 'all']:
        try:
            train_language_models(languages)
        except ImportError:
            logger.warning("Language model training module not available")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
