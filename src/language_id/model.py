"""
Language Identification Models for multilingual ASR.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageIdentifier:
    """Base class for language identification."""
    
    LANGUAGES = ['english', 'hindi', 'odia']
    LANGUAGE_CODES = {'english': 'en', 'hindi': 'hi', 'odia': 'or'}
    
    def predict(self, features: np.ndarray) -> str:
        """Predict language from MFCC features."""
        raise NotImplementedError
    
    def predict_proba(self, features: np.ndarray) -> dict:
        """Predict language probabilities."""
        raise NotImplementedError
    
    def get_code(self, language: str) -> str:
        """Get language code from language name."""
        return self.LANGUAGE_CODES.get(language, language)


class SVMLanguageIdentifier(LanguageIdentifier):
    """
    SVM-based language identification.
    
    Uses aggregated MFCC statistics (mean and variance)
    as features for classification.
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
        """
        self.model = SVC(kernel=kernel, C=C, probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"SVMLanguageIdentifier initialized: kernel={kernel}, C={C}")
    
    def _aggregate_features(self, features: np.ndarray) -> np.ndarray:
        """
        Aggregate temporal features into fixed-size vector.
        
        Args:
            features: MFCC features [T x D]
        
        Returns:
            Aggregated features [2*D] (mean + std)
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return np.concatenate([mean, std])
    
    def fit(
        self,
        X: List[np.ndarray],
        y: List[str]
    ) -> 'SVMLanguageIdentifier':
        """
        Train the language identifier.
        
        Args:
            X: List of MFCC feature matrices
            y: List of language labels
        
        Returns:
            self
        """
        # Aggregate features
        X_agg = np.array([self._aggregate_features(f) for f in X])
        
        # Convert labels to indices
        label_to_idx = {lang: i for i, lang in enumerate(self.LANGUAGES)}
        y_idx = np.array([label_to_idx[lang] for lang in y])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_agg)
        
        # Train SVM
        self.model.fit(X_scaled, y_idx)
        self.is_trained = True
        
        logger.info(f"Trained on {len(X)} samples")
        return self
    
    def predict(self, features: np.ndarray) -> str:
        """
        Predict language.
        
        Args:
            features: MFCC features [T x D]
        
        Returns:
            Language name
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_agg = self._aggregate_features(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X_agg)
        
        idx = self.model.predict(X_scaled)[0]
        return self.LANGUAGES[idx]
    
    def predict_proba(self, features: np.ndarray) -> dict:
        """
        Predict language probabilities.
        
        Args:
            features: MFCC features [T x D]
        
        Returns:
            Dictionary of language probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X_agg = self._aggregate_features(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X_agg)
        
        probs = self.model.predict_proba(X_scaled)[0]
        return {lang: prob for lang, prob in zip(self.LANGUAGES, probs)}
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SVMLanguageIdentifier':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        identifier = cls()
        identifier.model = data['model']
        identifier.scaler = data['scaler']
        identifier.is_trained = data['is_trained']
        
        logger.info(f"Model loaded from {path}")
        return identifier


class NeuralLanguageIdentifier(LanguageIdentifier):
    """
    Neural network-based language identification using LSTM.
    """
    
    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize neural language identifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralLanguageIdentifier")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = self._build_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"NeuralLanguageIdentifier initialized on {self.device}")
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM model."""
        class LIDModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.fc = nn.Linear(hidden_dim * 2, num_classes)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # x: [B, T, D]
                output, (hidden, _) = self.lstm(x)
                # Use last hidden state from both directions
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
                hidden = self.dropout(hidden)
                return self.fc(hidden)
        
        return LIDModel(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            len(self.LANGUAGES),
            self.dropout
        )
    
    def predict(self, features: np.ndarray) -> str:
        """Predict language."""
        probs = self.predict_proba(features)
        return max(probs, key=probs.get)
    
    def predict_proba(self, features: np.ndarray) -> dict:
        """Predict language probabilities."""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return {lang: float(prob) for lang, prob in zip(self.LANGUAGES, probs)}
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NeuralLanguageIdentifier':
        """Load model from file."""
        data = torch.load(path, map_location='cpu')
        
        identifier = cls(**data['config'])
        identifier.model.load_state_dict(data['model_state'])
        
        logger.info(f"Model loaded from {path}")
        return identifier
