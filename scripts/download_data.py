#!/usr/bin/env python
"""
Download datasets for Multilingual Indic ASR.
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


DATASETS = {
    'english': {
        'openslr': {
            'url': 'https://www.openslr.org/12',
            'description': 'LibriSpeech (OpenSLR) - Download "test-clean.tar.gz"',
            'manual': True
        }
    },
    'hindi': {
        'openslr': {
            'url': 'https://www.openslr.org/103',
            'description': 'OpenSLR Indic (Hindi) - Download "Hindi_test.tar.gz"',
            'manual': True
        }
    },
    'odia': {
        'openslr': {
            'url': 'https://www.openslr.org/103',
            'description': 'OpenSLR Indic (Odia) - Download "Odia_test.tar.gz"',
            'manual': True
        }
    }
}


def create_data_directories():
    """Create data directory structure."""
    base_path = Path('data')
    
    dirs = [
        'raw/english/clips',
        'raw/hindi/clips',
        'raw/odia/clips',
        'processed/english/train',
        'processed/english/val',
        'processed/english/test',
        'processed/hindi/train',
        'processed/hindi/val',
        'processed/hindi/test',
        'processed/odia/train',
        'processed/odia/val',
        'processed/odia/test',
        'features/english/train',
        'features/english/val',
        'features/english/test',
        'features/hindi/train',
        'features/hindi/val',
        'features/hindi/test',
        'features/odia/train',
        'features/odia/val',
        'features/odia/test',
        'transcripts/english',
        'transcripts/hindi',
        'transcripts/odia'
    ]
    
    for dir_path in dirs:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {base_path / dir_path}")


def print_download_instructions():
    """Print download instructions for each dataset."""
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    for language, sources in DATASETS.items():
        print(f"\n{'='*70}")
        print(f" {language.upper()}")
        print(f"{'='*70}")
        
        for name, info in sources.items():
            print(f"\n📦 {info['description']}")
            print(f"   URL: {info['url']}")
            print(f"   Target: data/raw/{language}/")
            if info.get('manual'):
                print("   ⚠️  Manual download required")
    
    print("\n" + "="*70)
    print("AFTER DOWNLOADING:")
    print("="*70)
    print("""
1. Extract audio files to: data/raw/<language>/clips/
2. Place transcript files in: data/raw/<language>/
3. Run: python scripts/prepare_data.py
    """)


def main():
    """Main entry point."""
    print("\n🎙️ Multilingual Indic ASR - Data Download Script")
    print("="*50)
    
    # Create directories
    print("\n📁 Creating data directories...")
    create_data_directories()
    
    # Print instructions
    print_download_instructions()
    
    print("\n✅ Directory structure created successfully!")
    print("📥 Please download datasets manually from the URLs above.")


if __name__ == "__main__":
    main()
