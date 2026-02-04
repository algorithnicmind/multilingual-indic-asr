"""
End-to-end ASR Inference Pipeline.

Combines all modules for complete speech-to-text transcription.
"""

import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .preprocessing import AudioPreprocessor
from .features import MFCCExtractor
from .language_id import SVMLanguageIdentifier
from .acoustic_model import AcousticModel
from .language_model import NGramLanguageModel
from .decoder import BeamSearchDecoder, GreedyDecoder

logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    Complete ASR pipeline for multilingual speech recognition.
    
    Pipeline:
        Audio -> Preprocess -> Features -> Language ID -> 
        Acoustic Model -> Decoder -> Text
    """
    
    LANGUAGES = ['english', 'hindi', 'odia']
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize ASR pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models_loaded = False
        
        # Initialize components
        self._init_components()
        
        logger.info("ASRPipeline initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Return default config
        return {
            'audio': {'sample_rate': 16000},
            'features': {
                'n_mfcc': 13,
                'use_delta': True,
                'use_delta_delta': True
            },
            'decoder': {
                'beam_width': 10,
                'lm_weight': 0.4
            }
        }
    
    def _init_components(self):
        """Initialize pipeline components."""
        # Audio preprocessor
        self.preprocessor = AudioPreprocessor({
            'sample_rate': self.config.get('audio', {}).get('sample_rate', 16000),
            'normalize': True,
            'remove_silence': True
        })
        
        # Feature extractor
        self.feature_extractor = MFCCExtractor(
            self.config.get('features', {})
        )
        
        # Placeholders for trained models
        self.language_identifier = None
        self.acoustic_models = {}
        self.language_models = {}
        self.decoders = {}
    
    def load_models(self, models_dir: str = 'models') -> None:
        """
        Load trained models from directory.
        
        Args:
            models_dir: Directory containing model files
        """
        models_path = Path(models_dir)
        
        # Load language identifier
        lid_path = models_path / 'language_id' / 'model.pkl'
        if lid_path.exists():
            self.language_identifier = SVMLanguageIdentifier.load(str(lid_path))
            logger.info("Loaded language identifier")
        
        # Load acoustic models
        for lang in self.LANGUAGES:
            am_path = models_path / 'acoustic' / lang / 'model.pt'
            if am_path.exists():
                self.acoustic_models[lang] = AcousticModel.load(str(am_path))
                logger.info(f"Loaded acoustic model for {lang}")
        
        # Load language models
        for lang in self.LANGUAGES:
            lm_path = models_path / 'language' / lang / 'model.pkl'
            if lm_path.exists():
                self.language_models[lang] = NGramLanguageModel.load(str(lm_path))
                logger.info(f"Loaded language model for {lang}")
        
        # Initialize decoders
        self._init_decoders()
        
        self.models_loaded = True
        logger.info("All models loaded successfully")
    
    def _init_decoders(self):
        """Initialize decoders for each language."""
        beam_width = self.config.get('decoder', {}).get('beam_width', 10)
        lm_weight = self.config.get('decoder', {}).get('lm_weight', 0.4)
        
        for lang in self.LANGUAGES:
            vocab = self._get_vocab(lang)
            lm = self.language_models.get(lang)
            
            self.decoders[lang] = BeamSearchDecoder(
                vocab=vocab,
                language_model=lm,
                beam_width=beam_width,
                lm_weight=lm_weight
            )
    
    def _get_vocab(self, language: str) -> list:
        """Get vocabulary for language."""
        # Placeholder - should load from file
        if language == 'english':
            return [''] + list('abcdefghijklmnopqrstuvwxyz ')
        elif language == 'hindi':
            # Hindi characters
            return [''] + list('अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह ')
        elif language == 'odia':
            # Odia characters
            return [''] + list('ଅଆଇଈଉଊଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହ ')
        return ['']
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language override (auto-detect if None)
        
        Returns:
            Dictionary with transcription results:
                - 'language': Detected/specified language
                - 'language_code': Language code (en/hi/or)
                - 'text': Transcribed text
                - 'confidence': Confidence score
        """
        # Preprocess audio
        audio = self.preprocessor.process(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract(audio)
        
        # Identify language (or use override)
        if language is None:
            if self.language_identifier is not None:
                language = self.language_identifier.predict(features)
            else:
                language = 'english'  # Default
        
        language_code = {
            'english': 'en',
            'hindi': 'hi',
            'odia': 'or'
        }.get(language, 'en')
        
        # Get transcription
        text = ""
        confidence = 0.0
        
        if language in self.acoustic_models and language in self.decoders:
            import torch
            
            # Run acoustic model
            model = self.acoustic_models[language]
            model.eval()
            
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0)
                log_probs = model(x)
            
            # Decode
            decoder = self.decoders[language]
            text = decoder.decode(log_probs[0])
            confidence = decoder.get_confidence()
        
        return {
            'language': language,
            'language_code': language_code,
            'text': text,
            'confidence': confidence
        }
    
    def transcribe_audio(
        self,
        audio_array,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio array to text.
        
        Args:
            audio_array: NumPy array of audio samples
            sample_rate: Sample rate of audio
            language: Optional language override
        
        Returns:
            Dictionary with transcription results
        """
        # Preprocess audio array
        audio = self.preprocessor.process(audio_array, sample_rate)
        
        # Extract features
        features = self.feature_extractor.extract(audio)
        
        # Identify language
        if language is None:
            if self.language_identifier is not None:
                language = self.language_identifier.predict(features)
            else:
                language = 'english'
        
        language_code = {
            'english': 'en',
            'hindi': 'hi',
            'odia': 'or'
        }.get(language, 'en')
        
        # For now, return placeholder (models not loaded)
        return {
            'language': language,
            'language_code': language_code,
            'text': f"[Transcription for {language}]",
            'confidence': 0.5
        }


def transcribe(audio_path: str, config_path: str = 'config.yaml') -> dict:
    """
    Convenience function for transcription.
    
    Args:
        audio_path: Path to audio file
        config_path: Path to configuration
    
    Returns:
        Transcription results dictionary
    """
    pipeline = ASRPipeline(config_path)
    pipeline.load_models()
    return pipeline.transcribe(audio_path)
