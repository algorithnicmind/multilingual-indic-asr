"""
Beam Search Decoder for CTC output with Language Model integration.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BeamHypothesis:
    """Represents a beam hypothesis."""
    prefix: str
    score: float
    lm_score: float = 0.0


class BeamSearchDecoder:
    """
    Beam search decoder with language model integration.
    
    Combines acoustic model scores with language model scores
    for better transcription quality.
    """
    
    def __init__(
        self,
        vocab: List[str],
        language_model=None,
        beam_width: int = 10,
        lm_weight: float = 0.4,
        blank_idx: int = 0
    ):
        """
        Initialize beam search decoder.
        
        Args:
            vocab: Vocabulary list
            language_model: Optional N-gram language model
            beam_width: Number of beams to keep
            lm_weight: Weight for language model scores
            blank_idx: Index of CTC blank token
        """
        self.vocab = vocab
        self.lm = language_model
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.blank_idx = blank_idx
        
        # Build index to token mapping
        self.idx_to_token = {i: token for i, token in enumerate(vocab)}
        
        self._best_score = 0.0
        
        logger.info(f"BeamSearchDecoder: beam_width={beam_width}, lm_weight={lm_weight}")
    
    def decode(
        self,
        log_probs: Union[np.ndarray, torch.Tensor]
    ) -> str:
        """
        Decode log probabilities using beam search.
        
        Args:
            log_probs: Log probabilities [time, vocab_size]
        
        Returns:
            Best decoded text string
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        
        T, V = log_probs.shape
        
        # Initialize beams
        beams = [BeamHypothesis(prefix='', score=0.0)]
        
        for t in range(T):
            candidates = []
            
            for beam in beams:
                for v in range(V):
                    token = self.idx_to_token.get(v, '')
                    am_score = log_probs[t, v]
                    
                    if v == self.blank_idx:
                        # Blank: keep prefix unchanged
                        new_prefix = beam.prefix
                    elif beam.prefix and beam.prefix[-1] == token:
                        # Repeated character: skip (handled by CTC)
                        continue
                    else:
                        # New character
                        new_prefix = beam.prefix + token
                    
                    # Calculate language model score
                    lm_score = self._get_lm_score(new_prefix)
                    
                    # Combined score
                    new_score = beam.score + am_score + self.lm_weight * lm_score
                    
                    candidates.append(BeamHypothesis(
                        prefix=new_prefix,
                        score=new_score,
                        lm_score=lm_score
                    ))
            
            # Merge same prefixes and keep top beams
            prefix_to_best = {}
            for candidate in candidates:
                if candidate.prefix not in prefix_to_best:
                    prefix_to_best[candidate.prefix] = candidate
                elif candidate.score > prefix_to_best[candidate.prefix].score:
                    prefix_to_best[candidate.prefix] = candidate
            
            beams = sorted(
                prefix_to_best.values(),
                key=lambda x: x.score,
                reverse=True
            )[:self.beam_width]
        
        # Return best hypothesis
        if beams:
            self._best_score = beams[0].score
            return beams[0].prefix
        return ''
    
    def _get_lm_score(self, prefix: str) -> float:
        """Calculate language model score for prefix."""
        if self.lm is None or len(prefix) < 2:
            return 0.0
        
        # Get last word for scoring
        words = prefix.split()
        if len(words) < 1:
            return 0.0
        
        try:
            if len(words) >= 3:
                context = tuple(words[-3:-1])
            elif len(words) == 2:
                context = (words[-2],)
            else:
                context = ()
            
            return self.lm.log_probability(words[-1], context)
        except:
            return 0.0
    
    def get_confidence(self) -> float:
        """Get confidence score of last decode."""
        return min(1.0, np.exp(self._best_score / 100))
    
    def decode_batch(
        self,
        log_probs: Union[np.ndarray, torch.Tensor]
    ) -> List[str]:
        """Decode batch of log probabilities."""
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        
        batch_size = log_probs.shape[0]
        return [self.decode(log_probs[i]) for i in range(batch_size)]
