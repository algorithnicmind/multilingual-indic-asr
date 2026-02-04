"""
Acoustic Model for speech recognition.

CNN + BiLSTM architecture with CTC loss.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AcousticModel(nn.Module):
    """
    CNN-BiLSTM acoustic model with CTC output.
    
    Architecture:
        Input (MFCC) -> CNN -> BiLSTM -> Linear -> LogSoftmax
    """
    
    def __init__(
        self,
        input_dim: int = 39,
        hidden_dim: int = 256,
        num_lstm_layers: int = 3,
        vocab_size: int = 100,
        dropout: float = 0.3
    ):
        """
        Initialize acoustic model.
        
        Args:
            input_dim: Input feature dimension (MFCC)
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            vocab_size: Output vocabulary size (phonemes + blank)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(0.1),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        
        # Calculate CNN output dimension
        cnn_output_dim = 128 * (input_dim // 2)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"AcousticModel: input={input_dim}, hidden={hidden_dim}, vocab={vocab_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, time, features]
        
        Returns:
            Log probabilities [batch, time', vocab_size]
        """
        batch_size, time_steps, features = x.size()
        
        # Reshape for CNN: [B, 1, T, F]
        x = x.unsqueeze(1)
        
        # CNN forward
        x = self.cnn(x)  # [B, C, T', F']
        
        # Reshape for LSTM: [B, T', C*F']
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        
        # LSTM forward
        x, _ = self.lstm(x)
        x = self.dropout(x)
        
        # Output projection
        x = self.fc(x)
        x = self.log_softmax(x)
        
        return x
    
    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate output sequence lengths after CNN pooling.
        
        Args:
            input_lengths: Input sequence lengths
        
        Returns:
            Output sequence lengths
        """
        # Account for CNN pooling (factor of 2)
        return input_lengths // 2
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'vocab_size': self.vocab_size
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'AcousticModel':
        """Load model from checkpoint."""
        data = torch.load(path, map_location=device)
        model = cls(**data['config'])
        model.load_state_dict(data['model_state'])
        logger.info(f"Model loaded from {path}")
        return model


class CTCLoss(nn.Module):
    """CTC Loss wrapper with proper handling."""
    
    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='mean', zero_infinity=True)
    
    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            log_probs: [batch, time, vocab] - Model outputs
            targets: [total_target_length] - Concatenated targets
            input_lengths: [batch] - Length of each input sequence
            target_lengths: [batch] - Length of each target sequence
        
        Returns:
            CTC loss value
        """
        # CTC expects [time, batch, vocab]
        log_probs = log_probs.permute(1, 0, 2)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
