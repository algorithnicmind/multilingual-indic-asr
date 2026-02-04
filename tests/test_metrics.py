"""
Tests for metrics module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import word_error_rate, character_error_rate, wer_details


class TestWordErrorRate:
    """Test WER calculation."""
    
    def test_identical_strings(self):
        """Test WER for identical strings."""
        wer = word_error_rate("hello world", "hello world")
        assert wer == 0.0
    
    def test_one_substitution(self):
        """Test WER with one substitution."""
        wer = word_error_rate("hello world", "hello earth")
        assert wer == 50.0  # 1/2 = 50%
    
    def test_one_deletion(self):
        """Test WER with one deletion."""
        wer = word_error_rate("hello world", "hello")
        assert wer == 50.0  # 1/2 = 50%
    
    def test_one_insertion(self):
        """Test WER with one insertion."""
        wer = word_error_rate("hello world", "hello beautiful world")
        assert wer == 50.0  # 1/2 = 50%
    
    def test_empty_reference(self):
        """Test WER with empty reference."""
        wer = word_error_rate("", "hello")
        assert wer == 100.0
    
    def test_empty_hypothesis(self):
        """Test WER with empty hypothesis."""
        wer = word_error_rate("hello world", "")
        assert wer == 100.0
    
    def test_both_empty(self):
        """Test WER with both empty."""
        wer = word_error_rate("", "")
        assert wer == 0.0


class TestCharacterErrorRate:
    """Test CER calculation."""
    
    def test_identical_strings(self):
        """Test CER for identical strings."""
        cer = character_error_rate("hello", "hello")
        assert cer == 0.0
    
    def test_one_error(self):
        """Test CER with one error."""
        cer = character_error_rate("hello", "hallo")
        assert cer == 20.0  # 1/5 = 20%


class TestWERDetails:
    """Test detailed WER calculation."""
    
    def test_details(self):
        """Test WER details."""
        details = wer_details("the cat sat", "a cat sit")
        
        assert 'wer' in details
        assert 'substitutions' in details
        assert 'deletions' in details
        assert 'insertions' in details
        assert details['reference_length'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
