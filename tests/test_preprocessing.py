"""
Tests for preprocessing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import AudioPreprocessor
from src.preprocessing.audio_utils import validate_audio


class TestAudioPreprocessor:
    """Test AudioPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'sample_rate': 16000,
            'normalize': True,
            'remove_silence': False,  # Disable for predictable output
            'pre_emphasis': 0.97
        }
        self.preprocessor = AudioPreprocessor(self.config)
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        assert self.preprocessor.sample_rate == 16000
        assert self.preprocessor.normalize == True
    
    def test_normalize(self):
        """Test audio normalization."""
        audio = np.array([0.5, -0.25, 0.75, -0.1])
        normalized = self.preprocessor._normalize(audio)
        
        # Should be normalized to [-1, 1]
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) == pytest.approx(1.0)
    
    def test_pre_emphasis(self):
        """Test pre-emphasis filter."""
        audio = np.array([1.0, 0.5, 0.25, 0.125])
        emphasized = self.preprocessor._pre_emphasis(audio)
        
        # Output should have same length as input
        assert len(emphasized) == len(audio)
        
        # First sample should be unchanged
        assert emphasized[0] == audio[0]
    
    def test_process_array(self):
        """Test processing audio array."""
        # Create test audio: 1 second of sine wave
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        processed = self.preprocessor.process(audio, sample_rate=sr)
        
        # Should return numpy array
        assert isinstance(processed, np.ndarray)
        
        # Should not be empty
        assert len(processed) > 0


class TestAudioValidation:
    """Test audio validation functions."""
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        is_valid, message = validate_audio("nonexistent.wav")
        assert is_valid == False
        assert "not found" in message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
