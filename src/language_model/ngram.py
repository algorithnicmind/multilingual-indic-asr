"""
N-gram Language Model for ASR decoding.
"""

import pickle
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NGramLanguageModel:
    """
    N-gram language model with Kneser-Ney smoothing.
    
    Supports unigram, bigram, and trigram models.
    """
    
    def __init__(self, n: int = 3, discount: float = 0.75):
        """
        Initialize N-gram model.
        
        Args:
            n: Order of the N-gram (1, 2, or 3)
            discount: Discount factor for Kneser-Ney smoothing
        """
        self.n = n
        self.discount = discount
        
        # N-gram counts
        self.ngram_counts = [defaultdict(int) for _ in range(n)]
        
        # Continuation counts for Kneser-Ney
        self.continuation_counts = [defaultdict(int) for _ in range(n)]
        
        # Vocabulary
        self.vocab = set()
        self.vocab_size = 0
        
        # Special tokens
        self.sos = '<s>'
        self.eos = '</s>'
        self.unk = '<unk>'
        
        logger.info(f"NGramLanguageModel initialized: n={n}, discount={discount}")
    
    def train(self, corpus: List[str]) -> None:
        """
        Train language model on text corpus.
        
        Args:
            corpus: List of sentences (strings)
        """
        logger.info(f"Training on {len(corpus)} sentences")
        
        for sentence in corpus:
            # Tokenize
            tokens = sentence.strip().lower().split()
            
            # Add to vocabulary
            self.vocab.update(tokens)
            
            # Add sentence markers
            tokens = [self.sos] * (self.n - 1) + tokens + [self.eos]
            
            # Count N-grams
            for i in range(len(tokens)):
                for order in range(min(self.n, i + 1)):
                    ngram = tuple(tokens[i - order:i + 1])
                    self.ngram_counts[order][ngram] += 1
                    
                    # Track continuation
                    if order > 0:
                        prefix = ngram[:-1]
                        self.continuation_counts[order][prefix] += 1
        
        self.vocab_size = len(self.vocab) + 3  # +3 for sos, eos, unk
        logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def probability(self, word: str, context: Tuple[str, ...] = None) -> float:
        """
        Calculate P(word | context) with Kneser-Ney smoothing.
        
        Args:
            word: Target word
            context: Context words as tuple
        
        Returns:
            Probability P(word | context)
        """
        context = context or ()
        
        # Ensure context length is at most n-1
        context = context[-(self.n - 1):]
        
        return self._kneser_ney_prob(word, context)
    
    def _kneser_ney_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Kneser-Ney smoothed probability."""
        ngram = context + (word,)
        order = len(ngram) - 1
        
        if order == 0:
            # Unigram
            count = self.ngram_counts[0].get((word,), 0)
            total = sum(self.ngram_counts[0].values())
            return (count + 1) / (total + self.vocab_size)  # Add-one smoothing
        
        # Higher order N-gram
        prefix_count = self.ngram_counts[order - 1].get(context, 0)
        
        if prefix_count == 0:
            # Back off to lower order
            return self._kneser_ney_prob(word, context[1:])
        
        ngram_count = self.ngram_counts[order].get(ngram, 0)
        
        # Discounted probability
        prob = max(ngram_count - self.discount, 0) / prefix_count
        
        # Interpolation weight
        num_continuations = len([ng for ng in self.ngram_counts[order] 
                                 if ng[:-1] == context])
        lambda_weight = self.discount * num_continuations / prefix_count
        
        # Recursive backoff
        prob += lambda_weight * self._kneser_ney_prob(word, context[1:])
        
        return prob
    
    def log_probability(self, word: str, context: Tuple[str, ...] = None) -> float:
        """Calculate log probability."""
        prob = self.probability(word, context)
        return math.log(prob + 1e-10)
    
    def score_sentence(self, sentence: str) -> float:
        """
        Calculate log probability of a sentence.
        
        Args:
            sentence: Input sentence
        
        Returns:
            Log probability of the sentence
        """
        tokens = sentence.strip().lower().split()
        tokens = [self.sos] * (self.n - 1) + tokens + [self.eos]
        
        log_prob = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            log_prob += self.log_probability(word, context)
        
        return log_prob
    
    def perplexity(self, corpus: List[str]) -> float:
        """
        Calculate perplexity on a corpus.
        
        Args:
            corpus: List of sentences
        
        Returns:
            Perplexity score
        """
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in corpus:
            total_log_prob += self.score_sentence(sentence)
            total_words += len(sentence.split()) + 1  # +1 for eos
        
        avg_log_prob = total_log_prob / total_words
        return math.exp(-avg_log_prob)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'n': self.n,
            'discount': self.discount,
            'ngram_counts': [dict(c) for c in self.ngram_counts],
            'continuation_counts': [dict(c) for c in self.continuation_counts],
            'vocab': list(self.vocab),
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NGramLanguageModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'], discount=data['discount'])
        model.ngram_counts = [defaultdict(int, c) for c in data['ngram_counts']]
        model.continuation_counts = [defaultdict(int, c) for c in data['continuation_counts']]
        model.vocab = set(data['vocab'])
        model.vocab_size = data['vocab_size']
        
        logger.info(f"Model loaded from {path}")
        return model
