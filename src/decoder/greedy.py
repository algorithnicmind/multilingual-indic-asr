"""
Greedy Decoder for CTC output.
"""

import numpy as np
import torch
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class GreedyDecoder:
    """
    Simple greedy CTC decoder.
    
    Selects the most probable token at each time step,
    then removes blanks and repeated tokens.
    """
    
    def __init__(self, vocab: List[str], blank_idx: int = 0):
        """
        Initialize greedy decoder.
        
        Args:
            vocab: Vocabulary list (index to token mapping)
            blank_idx: Index of CTC blank token
        """
        self.vocab = vocab
        self.blank_idx = blank_idx
        self.vocab_size = len(vocab)
        
        logger.info(f"GreedyDecoder initialized: vocab_size={self.vocab_size}")
    
    def decode(
        self,
        log_probs: Union[np.ndarray, torch.Tensor]
    ) -> str:
        """
        Decode log probabilities to text.
        
        Args:
            log_probs: Log probabilities [time, vocab_size]
        
        Returns:
            Decoded text string
        """
        # Convert to numpy if tensor
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        
        # Get most probable tokens
        best_path = np.argmax(log_probs, axis=-1)
        
        # Remove blanks and repeated tokens
        decoded = []
        prev_token = None
        
        for idx in best_path:
            if idx != self.blank_idx and idx != prev_token:
                if idx < len(self.vocab):
                    decoded.append(self.vocab[idx])
            prev_token = idx
        
        return ''.join(decoded)
    
    def decode_batch(
        self,
        log_probs: Union[np.ndarray, torch.Tensor]
    ) -> List[str]:
        """
        Decode batch of log probabilities.
        
        Args:
            log_probs: Log probabilities [batch, time, vocab_size]
        
        Returns:
            List of decoded text strings
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        
        batch_size = log_probs.shape[0]
        return [self.decode(log_probs[i]) for i in range(batch_size)]
