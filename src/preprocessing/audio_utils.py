"""
Audio utility functions for loading, saving, and validating audio files.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_audio(
    audio_path: str,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and convert to standard format.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
        mono: Convert to mono (default: True)
        normalize: Normalize amplitude (default: True)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=mono)
    
    # Normalize amplitude
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    logger.debug(f"Loaded audio: {audio_path}, duration: {len(audio)/sr:.2f}s")
    return audio, sr


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 16000
) -> None:
    """
    Save audio array to WAV file.
    
    Args:
        audio: Audio samples array
        output_path: Output file path
        sample_rate: Sample rate (default: 16000)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as WAV
    sf.write(output_path, audio, sample_rate)
    logger.debug(f"Saved audio to: {output_path}")


def validate_audio(
    audio_path: str,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    min_amplitude: float = 0.01
) -> Tuple[bool, str]:
    """
    Validate audio file meets requirements.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        min_amplitude: Minimum peak amplitude
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Check file exists
    if not Path(audio_path).exists():
        return False, f"File not found: {audio_path}"
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Check duration
        duration = len(audio) / sr
        if duration < min_duration:
            return False, f"Audio too short: {duration:.2f}s < {min_duration}s"
        if duration > max_duration:
            return False, f"Audio too long: {duration:.2f}s > {max_duration}s"
        
        # Check amplitude (not silence)
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < min_amplitude:
            return False, f"Audio is silent: max amplitude {max_amplitude:.4f}"
        
        return True, f"Valid audio: {duration:.2f}s"
        
    except Exception as e:
        return False, f"Error loading audio: {str(e)}"


def get_audio_info(audio_path: str) -> dict:
    """
    Get audio file information.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with audio metadata
    """
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    
    return {
        'path': audio_path,
        'sample_rate': sr,
        'duration': len(audio) / sr,
        'samples': len(audio),
        'max_amplitude': float(np.max(np.abs(audio))),
        'mean_amplitude': float(np.mean(np.abs(audio))),
        'rms': float(np.sqrt(np.mean(audio ** 2)))
    }
