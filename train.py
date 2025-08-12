"""
Training script for SQuAD Transformer
Usage: python train.py
"""

import os
import sys
import torch
from datetime import datetime

# Import your transformer model
from transformer_qa import (
    TransformerQA, 
    SQuADDataset, 
    train_model, 
    check_gpu,
    DEVICE
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def main():
    # Configuration
    CONFIG = {
        'batch_size': 16,           # Reduce to 8 if GPU memory issues
        'epochs': 3,
        'learning_rate': 3e-5,
        'max_len': 384,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'train_file': 'train-v2.0.json',
        'dev_file': 'dev-v2.0.json',
        'output_dir': 'outputs'
    }
    
    print("="*60)
    print("SQuAD TRANSFORMER TRAINING")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Check files exist
    if not os.path.exists(CONFIG['train_file']):
        print(f"ERROR: Training file '{CONFIG['train_file']}' not found!")
        print("Please ensure train.json is in the current directory")
        sys.exit(1)
    
    if not os.path.exists(CONFIG['dev_file']):
        print(f"ERROR: Dev file '{CONFIG['dev_file']}' not found!")
        print("Please ensure dev.json is in the current directory")
        sys.exit(1)
    
    # Initialize tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load datasets
    print("\n[2/5] Loading training dataset...")
    train_dataset = SQuADDataset(
        CONFIG['train_file'], 
        tokenizer, 
        max_len=CONFIG['max_len']
    )
    
    print("\n[3/5] Loading validation dataset...")
    val_dataset = SQuADDataset(
        CONFIG['dev_file'], 
        tokenizer, 
        max_len=CONFIG['max_len']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n[4/5] Initializing model...")
    model = TransformerQA(
        vocab_size=tokenizer.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    )
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Train model
    print("\n[5/5] Starting training...")
    print("="*60)
    train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=CONFIG['epochs'], 
        lr=CONFIG['learning_rate']
    )
    
    # Save model
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    model_path = os.path.join(
        CONFIG['output_dir'], 
        f"transformer_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'vocab_size': tokenizer.vocab_size
    }, model_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()