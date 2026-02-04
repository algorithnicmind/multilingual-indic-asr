"""
Tests for feature extraction module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import MFCCExtractor


class TestMFCCExtractor:
    """Test MFCCExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'sample_rate': 16000,
            'n_mfcc': 13,
            'use_delta': True,
            'use_delta_delta': True,
            'normalize_features': True
        }
        self.extractor = MFCCExtractor(self.config)
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.n_mfcc == 13
        assert self.extractor.sample_rate == 16000
    
    def test_feature_dim(self):
        """Test feature dimension property."""
        # 13 MFCCs + 13 deltas + 13 delta-deltas = 39
        assert self.extractor.feature_dim == 39
    
    def test_extract(self):
        """Test feature extraction."""
        # Create 1 second of test audio
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        
        features = self.extractor.extract(audio)
        
        # Should return 2D array
        assert features.ndim == 2
        
        # Should have 39 features (13 + 13 + 13)
        assert features.shape[1] == 39
        
        # Should have approximately 100 frames per second
        # (10ms hop = 100 frames/sec)
        expected_frames = sr // 160  # hop_length = 160
        assert abs(features.shape[0] - expected_frames) < 10
    
    def test_extract_short_audio(self):
        """Test extraction on short audio."""
        # 100ms of audio
        sr = 16000
        audio = np.random.randn(int(sr * 0.1)).astype(np.float32)
        
        features = self.extractor.extract(audio)
        
        # Should still produce output
        assert features.shape[0] > 0
        assert features.shape[1] == 39
    
    def test_normalization(self):
        """Test feature normalization."""
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        
        features = self.extractor.extract(audio)
        
        # Normalized features should have mean near 0
        mean = np.mean(features, axis=0)
        assert np.allclose(mean, 0, atol=0.1)


class TestMFCCWithoutDeltas:
    """Test MFCC extraction without delta features."""
    
    def test_no_deltas(self):
        """Test extraction without deltas."""
        config = {
            'n_mfcc': 13,
            'use_delta': False,
            'use_delta_delta': False
        }
        extractor = MFCCExtractor(config)
        
        assert extractor.feature_dim == 13
        
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract(audio)
        
        assert features.shape[1] == 13


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
