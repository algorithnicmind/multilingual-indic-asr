"""
Training script for Language Models.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.language_model.ngram import NGramLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text_corpus(language: str):
    """Load text corpus for language model training."""
    # Load from transcripts
    transcript_file = Path('data/transcripts') / language / 'train.txt'
    
    sentences = []
    if transcript_file.exists():
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sentences.append(parts[1])
    
    return sentences


def train_language_model(language: str, n: int = 3):
    """Train language model for a language."""
    logger.info(f"Training {n}-gram language model for {language}")
    
    # Load corpus
    sentences = load_text_corpus(language)
    
    if len(sentences) == 0:
        logger.warning(f"No training data for {language}")
        logger.info("Please run scripts/prepare_data.py first.")
        return None
    
    logger.info(f"Training on {len(sentences)} sentences")
    
    # Train model
    model = NGramLanguageModel(n=n, discount=0.75)
    model.train(sentences)
    
    logger.info(f"Vocabulary size: {model.vocab_size}")
    
    # Calculate perplexity on training data
    perplexity = model.perplexity(sentences[:1000])  # Sample
    logger.info(f"Training perplexity: {perplexity:.2f}")
    
    # Save model
    model_dir = Path('models/language') / language
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / 'model.pkl'))
    
    logger.info(f"Model saved to {model_dir / 'model.pkl'}")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()
    
    train_language_model(args.language, args.n)
