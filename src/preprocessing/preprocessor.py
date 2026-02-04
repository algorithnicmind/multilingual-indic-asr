"""
Audio Preprocessor for preparing audio files for ASR.
"""

import numpy as np
import librosa
from typing import Union, Optional
import logging

from .audio_utils import load_audio

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    Preprocessor for audio signals.
    
    Handles:
    - Loading and resampling
    - Mono conversion
    - Amplitude normalization
    - Silence removal
    - Pre-emphasis filtering
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        config = config or {}
        
        self.sample_rate = config.get('sample_rate', 16000)
        self.normalize = config.get('normalize', True)
        self.remove_silence = config.get('remove_silence', True)
        self.silence_threshold = config.get('silence_threshold', 0.01)
        self.silence_min_duration = config.get('silence_min_duration', 0.1)
        self.pre_emphasis_coef = config.get('pre_emphasis', 0.97)
        
        logger.info(f"AudioPreprocessor initialized: sr={self.sample_rate}")
    
    def process(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Process audio input.
        
        Args:
            audio_input: Path to audio file or audio array
            sample_rate: Sample rate if audio_input is array
        
        Returns:
            Processed audio array
        """
        # Load audio if path is provided
        if isinstance(audio_input, str):
            audio, sr = load_audio(
                audio_input,
                target_sr=self.sample_rate,
                normalize=False
            )
        else:
            audio = audio_input
            sr = sample_rate or self.sample_rate
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=self.sample_rate
                )
        
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Remove silence
        if self.remove_silence:
            audio = self._remove_silence(audio)
        
        # Normalize amplitude
        if self.normalize:
            audio = self._normalize(audio)
        
        # Apply pre-emphasis
        audio = self._pre_emphasis(audio)
        
        logger.debug(f"Processed audio: {len(audio)} samples")
        return audio
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter."""
        return np.append(
            audio[0], 
            audio[1:] - self.pre_emphasis_coef * audio[:-1]
        )
    
    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove leading and trailing silence."""
        # Compute energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        hop_length = int(0.010 * self.sample_rate)    # 10ms
        
        # Simple energy-based VAD
        energy = np.array([
            np.sum(audio[i:i+frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop_length)
        ])
        
        if len(energy) == 0:
            return audio
        
        # Normalize energy
        energy = energy / (np.max(energy) + 1e-10)
        
        # Find speech regions
        threshold = self.silence_threshold
        speech_frames = np.where(energy > threshold)[0]
        
        if len(speech_frames) == 0:
            return audio
        
        # Get boundaries
        start_frame = max(0, speech_frames[0] - 5)
        end_frame = min(len(energy) - 1, speech_frames[-1] + 5)
        
        start_sample = start_frame * hop_length
        end_sample = min(len(audio), end_frame * hop_length + frame_length)
        
        return audio[start_sample:end_sample]


def preprocess_audio(audio_path: str, config: dict = None) -> np.ndarray:
    """
    Convenience function to preprocess an audio file.
    
    Args:
        audio_path: Path to audio file
        config: Configuration dictionary
    
    Returns:
        Processed audio array
    """
    preprocessor = AudioPreprocessor(config)
    return preprocessor.process(audio_path)
