"""
MFCC Feature Extractor for speech recognition.
"""

import numpy as np
import librosa
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """
    Extract MFCC features from audio signals.
    
    Features include:
    - 13 MFCC coefficients
    - 13 Delta coefficients (velocity)
    - 13 Delta-Delta coefficients (acceleration)
    
    Total: 39 features per frame
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize MFCC extractor.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        config = config or {}
        
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 13)
        self.n_fft = config.get('n_fft', 512)
        self.hop_length = config.get('hop_length', 160)  # 10ms
        self.win_length = config.get('win_length', 400)  # 25ms
        self.n_mels = config.get('n_mels', 40)
        self.fmin = config.get('fmin', 20)
        self.fmax = config.get('fmax', 8000)
        self.use_delta = config.get('use_delta', True)
        self.use_delta_delta = config.get('use_delta_delta', True)
        self.normalize_features = config.get('normalize_features', True)
        
        logger.info(f"MFCCExtractor initialized: n_mfcc={self.n_mfcc}")
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio samples (1D numpy array)
        
        Returns:
            features: MFCC features [T x D] where D = 13/26/39
        """
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        ).T  # Transpose to [T x n_mfcc]
        
        features = mfccs
        
        # Compute delta features
        if self.use_delta:
            delta = librosa.feature.delta(mfccs.T).T
            features = np.concatenate([features, delta], axis=1)
        
        # Compute delta-delta features
        if self.use_delta_delta:
            delta2 = librosa.feature.delta(mfccs.T, order=2).T
            features = np.concatenate([features, delta2], axis=1)
        
        # Normalize features (CMVN)
        if self.normalize_features:
            features = self._normalize(features)
        
        logger.debug(f"Extracted features: shape={features.shape}")
        return features
    
    def extract_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            features: MFCC features [T x D]
        """
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.extract(audio)
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Cepstral Mean and Variance Normalization (CMVN).
        
        Args:
            features: Feature matrix [T x D]
        
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-10
        return (features - mean) / std
    
    @property
    def feature_dim(self) -> int:
        """Get output feature dimension."""
        dim = self.n_mfcc
        if self.use_delta:
            dim += self.n_mfcc
        if self.use_delta_delta:
            dim += self.n_mfcc
        return dim


def extract_mfcc(audio: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Convenience function to extract MFCC features.
    
    Args:
        audio: Audio samples
        config: Configuration dictionary
    
    Returns:
        MFCC features [T x D]
    """
    extractor = MFCCExtractor(config)
    return extractor.extract(audio)
