"""
Training script for Acoustic Models.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.acoustic_model.model import AcousticModel, CTCLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """Dataset for acoustic model training."""
    
    def __init__(self, language: str, split: str = 'train'):
        self.language = language
        self.split = split
        
        self.features_dir = Path('data/features') / language / split
        self.transcripts = self._load_transcripts()
        
        self.vocab = self._build_vocab()
        self.samples = list(self.transcripts.keys())
    
    def _load_transcripts(self):
        """Load transcripts."""
        transcript_file = Path('data/transcripts') / self.language / f"{self.split}.txt"
        
        transcripts = {}
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        transcripts[parts[0]] = parts[1]
        
        return transcripts
    
    def _build_vocab(self):
        """Build character vocabulary."""
        chars = set()
        for text in self.transcripts.values():
            chars.update(list(text.lower()))
        
        vocab = ['<blank>'] + sorted(list(chars))
        return {c: i for i, c in enumerate(vocab)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load features
        feature_path = self.features_dir / f"{sample_id}.npy"
        if feature_path.exists():
            features = np.load(feature_path)
        else:
            features = np.zeros((100, 39))  # Placeholder
        
        # Get transcript and encode
        text = self.transcripts.get(sample_id, "")
        target = [self.vocab.get(c, 0) for c in text.lower()]
        
        return {
            'features': torch.FloatTensor(features),
            'target': torch.LongTensor(target),
            'input_length': len(features),
            'target_length': len(target)
        }


def collate_fn(batch):
    """Collate function for variable length sequences."""
    # Sort by input length (descending)
    batch = sorted(batch, key=lambda x: x['input_length'], reverse=True)
    
    # Pad features
    max_input_len = max(item['input_length'] for item in batch)
    max_target_len = max(item['target_length'] for item in batch)
    
    features = torch.zeros(len(batch), max_input_len, batch[0]['features'].size(-1))
    targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    input_lengths = []
    target_lengths = []
    
    for i, item in enumerate(batch):
        feat = item['features']
        tgt = item['target']
        features[i, :feat.size(0)] = feat
        targets[i, :tgt.size(0)] = tgt
        input_lengths.append(item['input_length'])
        target_lengths.append(item['target_length'])
    
    return {
        'features': features,
        'targets': targets,
        'input_lengths': torch.LongTensor(input_lengths),
        'target_lengths': torch.LongTensor(target_lengths)
    }


import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_acoustic_model(language: str, epochs: int = None, batch_size: int = None):
    config = load_config()
    if epochs is None:
        epochs = config['acoustic_model']['training']['epochs']
    if batch_size is None:
        batch_size = config['acoustic_model']['training']['batch_size']
    """Train acoustic model for a language."""
    logger.info(f"Training acoustic model for {language}")
    
    # Create dataset
    try:
        train_dataset = ASRDataset(language, 'train')
        val_dataset = ASRDataset(language, 'val')
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        logger.info("Please run scripts/prepare_data.py first.")
        return None
    
    if len(train_dataset) == 0:
        logger.warning(f"No training data for {language}")
        return None
    
    vocab_size = len(train_dataset.vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=collate_fn
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_dir = Path('models/acoustic') / language
    model_path = model_dir / 'model.pt'
    
    model = AcousticModel(
        input_dim=39,
        hidden_dim=64, # Matches new config
        num_lstm_layers=1, # Matches new config
        vocab_size=vocab_size,
        dropout=0.1
    ).to(device)
    
    if model_path.exists():
        logger.info(f"Resuming from checkpoint: {model_path}")
        try:
             model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
             logger.warning(f"Could not load checkpoint (might be architecture mismatch): {e}")
    
    # Loss and optimizer
    criterion = CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = batch['features'].to(device)
            targets = batch['targets']
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']
            
            # Account for CNN pooling
            output_lengths = model.get_output_lengths(input_lengths)
            
            optimizer.zero_grad()
            
            # Forward pass
            log_probs = model(features)
            
            # Flatten targets
            targets_flat = targets[targets != 0]
            
            # CTC loss
            loss = criterion(
                log_probs,
                targets_flat,
                output_lengths,
                target_lengths
            )
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_dir = Path('models/acoustic') / language
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(model_dir / 'model.pt'))
            logger.info(f"Saved best model (loss={best_loss:.4f})")
    
    logger.info(f"Training complete for {language}")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_acoustic_model(args.language, args.epochs)
