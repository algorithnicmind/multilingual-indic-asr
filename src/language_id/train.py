"""
Training script for Language Identification model.
"""

import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.language_id.model import SVMLanguageIdentifier
from src.features import MFCCExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LANGUAGES = ['english', 'hindi', 'odia']


def load_features(language: str, split: str = 'train'):
    """Load features for a language."""
    features_dir = Path('data/features') / language / split
    
    if not features_dir.exists():
        logger.warning(f"Features directory not found: {features_dir}")
        return []
    
    features = []
    for feature_file in features_dir.glob('*.npy'):
        feat = np.load(feature_file)
        features.append(feat)
    
    return features


def train_lid_model():
    """Train the language identification model."""
    logger.info("Loading training data...")
    
    X_train = []
    y_train = []
    
    for language in LANGUAGES:
        features = load_features(language, 'train')
        X_train.extend(features)
        y_train.extend([language] * len(features))
        logger.info(f"  {language}: {len(features)} samples")
    
    if len(X_train) == 0:
        logger.error("No training data found!")
        logger.info("Please run scripts/prepare_data.py first.")
        return None
    
    logger.info(f"Total training samples: {len(X_train)}")
    
    # Train model
    logger.info("Training SVM classifier...")
    model = SVMLanguageIdentifier(kernel='rbf', C=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    X_val = []
    y_val = []
    
    for language in LANGUAGES:
        features = load_features(language, 'val')
        X_val.extend(features)
        y_val.extend([language] * len(features))
    
    if len(X_val) > 0:
        correct = 0
        for feat, label in zip(X_val, y_val):
            pred = model.predict(feat)
            if pred == label:
                correct += 1
        
        accuracy = correct / len(X_val)
        logger.info(f"Validation Accuracy: {accuracy:.2%}")
    
    # Save model
    model_dir = Path('models/language_id')
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / 'model.pkl'))
    
    logger.info(f"Model saved to {model_dir / 'model.pkl'}")
    return model


if __name__ == "__main__":
    train_lid_model()
