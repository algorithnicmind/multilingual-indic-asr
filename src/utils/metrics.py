"""
Evaluation metrics for ASR.
"""

from typing import List, Tuple
import numpy as np


def edit_distance(s1: List[str], s2: List[str]) -> Tuple[int, int, int, int]:
    """
    Calculate edit distance with operation counts.
    
    Args:
        s1: Reference sequence
        s2: Hypothesis sequence
    
    Returns:
        Tuple of (distance, substitutions, deletions, insertions)
    """
    m, n = len(s1), len(s2)
    
    # Create DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = ('D', i, 0, 0)  # deletion
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = ('I', 0, 0, j)  # insertion
    
    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                insertion = dp[i][j - 1] + 1
                
                min_op = min(substitution, deletion, insertion)
                dp[i][j] = min_op
                
                # Track operations
                prev_s, prev_d, prev_i = 0, 0, 0
                if min_op == substitution:
                    if ops[i - 1][j - 1]:
                        _, prev_s, prev_d, prev_i = ops[i - 1][j - 1]
                    ops[i][j] = ('S', prev_s + 1, prev_d, prev_i)
                elif min_op == deletion:
                    if ops[i - 1][j]:
                        _, prev_s, prev_d, prev_i = ops[i - 1][j]
                    ops[i][j] = ('D', prev_s, prev_d + 1, prev_i)
                else:
                    if ops[i][j - 1]:
                        _, prev_s, prev_d, prev_i = ops[i][j - 1]
                    ops[i][j] = ('I', prev_s, prev_d, prev_i + 1)
    
    if ops[m][n]:
        _, subs, dels, ins = ops[m][n]
    else:
        subs, dels, ins = 0, 0, 0
    
    return dp[m][n], subs, dels, ins


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N * 100%
    
    Where:
        S = Substitutions
        D = Deletions
        I = Insertions
        N = Total words in reference
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
    
    Returns:
        WER as percentage (0-100)
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0
    
    distance, _, _, _ = edit_distance(ref_words, hyp_words)
    
    return (distance / len(ref_words)) * 100


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
    
    Returns:
        CER as percentage (0-100)
    """
    ref_chars = list(reference.strip().lower().replace(' ', ''))
    hyp_chars = list(hypothesis.strip().lower().replace(' ', ''))
    
    if len(ref_chars) == 0:
        return 100.0 if len(hyp_chars) > 0 else 0.0
    
    distance, _, _, _ = edit_distance(ref_chars, hyp_chars)
    
    return (distance / len(ref_chars)) * 100


def wer_details(reference: str, hypothesis: str) -> dict:
    """
    Get detailed WER statistics.
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
    
    Returns:
        Dictionary with detailed statistics
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()
    
    if len(ref_words) == 0:
        return {
            'wer': 100.0 if len(hyp_words) > 0 else 0.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': len(hyp_words),
            'reference_length': 0,
            'hypothesis_length': len(hyp_words)
        }
    
    distance, subs, dels, ins = edit_distance(ref_words, hyp_words)
    
    return {
        'wer': (distance / len(ref_words)) * 100,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'reference_length': len(ref_words),
        'hypothesis_length': len(hyp_words)
    }


def evaluate_batch(
    references: List[str],
    hypotheses: List[str]
) -> dict:
    """
    Evaluate a batch of transcriptions.
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
    
    Returns:
        Dictionary with aggregate statistics
    """
    assert len(references) == len(hypotheses)
    
    wer_scores = []
    cer_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        wer_scores.append(word_error_rate(ref, hyp))
        cer_scores.append(character_error_rate(ref, hyp))
    
    return {
        'wer_mean': np.mean(wer_scores),
        'wer_std': np.std(wer_scores),
        'wer_min': np.min(wer_scores),
        'wer_max': np.max(wer_scores),
        'cer_mean': np.mean(cer_scores),
        'cer_std': np.std(cer_scores),
        'num_samples': len(references)
    }
