import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GPU CHECK - CRITICAL FOR PERFORMANCE
# ============================================
def check_gpu():
    """Debug function to ensure GPU usage"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU AVAILABLE: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Current GPU: {torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        print("✗ WARNING: GPU NOT AVAILABLE - Training will be VERY slow!")
        print("  Please check CUDA installation")
    return device

# Initialize device globally
DEVICE = check_gpu()

# ============================================
# POSITIONAL ENCODING
# ============================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention is All You Need'"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term with proper scaling
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ============================================
# MULTI-HEAD ATTENTION
# ============================================
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with proper scaling"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Initialize weights with Xavier uniform
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output

# ============================================
# FEED FORWARD NETWORK
# ============================================
class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
    
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

# ============================================
# TRANSFORMER ENCODER LAYER
# ============================================
class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with residual connections and layer norm"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

# ============================================
# MAIN TRANSFORMER MODEL FOR QA
# ============================================
class TransformerQA(nn.Module):
    """Complete Transformer model for extractive question answering"""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, 
                 d_ff=1024, max_len=384, dropout=0.15):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Segment embeddings for question/context separation
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # Token type embeddings (question vs context)
        self.token_type_embedding = nn.Embedding(2, d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # QA-specific heads
        self.qa_outputs = nn.Linear(d_model, 2)  # start and end logits
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.qa_outputs.weight)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embedding(input_ids)
        
        # Add segment embeddings if provided
        if token_type_ids is not None:
            x = x + self.token_type_embedding(token_type_ids)
        
        # Scale embeddings (important for stability)
        x = x * math.sqrt(x.size(-1))
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # QA outputs
        logits = self.qa_outputs(x)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        
        return start_logits, end_logits

# ============================================
# DATASET CLASS
# ============================================
class SQuADDataset(Dataset):
    """Dataset class for SQuAD with proper preprocessing"""
    def __init__(self, data_path, tokenizer, max_len=384, stride=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.features = []
        
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self._preprocess(data['data'])
        print(f"Created {len(self.features)} training examples")
    
    def _preprocess(self, data):
        """Convert SQuAD data to features"""
        for article in tqdm(data, desc="Processing articles"):
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    question = qa['question']
                    qid = qa['id']
                    
                    # Handle both SQuAD 1.1 and 2.0
                    if 'is_impossible' in qa and qa['is_impossible']:
                        continue  # Skip unanswerable questions for now
                    
                    if not qa['answers']:
                        continue
                    
                    # Get answer (use first one for training)
                    answer = qa['answers'][0]
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)
                    
                    # Tokenize with sliding window
                    self._create_features(
                        question, context, answer_start, answer_end, qid
                    )
    
    def _create_features(self, question, context, answer_start, answer_end, qid):
        """Create features with sliding window for long contexts"""
        # Tokenize question and context
        tokenized = self.tokenizer(
            question,
            context,
            truncation='only_second',
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length'
        )
        
        # Process each window
        for i in range(len(tokenized['input_ids'])):
            input_ids = tokenized['input_ids'][i]
            attention_mask = tokenized['attention_mask'][i]
            token_type_ids = tokenized['token_type_ids'][i] if 'token_type_ids' in tokenized else None
            offset_mapping = tokenized['offset_mapping'][i]
            
            # Find token positions for answer
            cls_index = 0
            
            # Find where context starts (after [SEP])
            context_start = 1
            for idx, tid in enumerate(token_type_ids if token_type_ids else [0]*len(input_ids)):
                if tid == 1:
                    context_start = idx
                    break
            
            # Find answer token positions
            start_position = 0
            end_position = 0
            
            if answer_start >= 0:  # Has answer
                for idx, (start, end) in enumerate(offset_mapping):
                    if idx < context_start:
                        continue
                    
                    if start == answer_start:
                        start_position = idx
                    if end == answer_end:
                        end_position = idx
                        break
                
                # Skip if answer not in this window
                if start_position == 0 and end_position == 0:
                    continue
            
            self.features.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids if token_type_ids else [0]*len(input_ids), dtype=torch.long),
                'start_position': torch.tensor(start_position, dtype=torch.long),
                'end_position': torch.tensor(end_position, dtype=torch.long),
                'qid': qid
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

# ============================================
# TRAINING FUNCTION
# ============================================
def train_model(model, train_loader, val_loader, epochs=3, lr=5e-5):
    """Training loop with optimization techniques"""
    print(f"\n{'='*50}")
    print(f"TRAINING ON: {DEVICE}")
    print(f"{'='*50}\n")
    
    # Move model to GPU
    model = model.to(DEVICE)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to GPU - CRITICAL!
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            start_positions = batch['start_position'].to(DEVICE)
            end_positions = batch['end_position'].to(DEVICE)
            
            # GPU check every 100 batches
            if batch_idx % 100 == 0:
                assert input_ids.is_cuda, "ERROR: Tensors not on GPU!"
            
            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            
            # Calculate loss
            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")

def evaluate(model, data_loader, criterion):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            start_positions = batch['start_position'].to(DEVICE)
            end_positions = batch['end_position'].to(DEVICE)
            
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            
            start_loss = criterion(start_logits, start_positions)
            end_loss = criterion(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Hyperparameters
    BATCH_SIZE = 16  # Reduce if GPU memory issues
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    MAX_LEN = 384
    
    # Model parameters
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.1
    
    print("Initializing tokenizer...")
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    # Prefer local path so Transformers never tries the internet:
    #LOCAL_TOK_DIR = "/mnt/c/Users/vince/Desktop/hf_cache/bert-base-uncased"
    LOCAL_TOK_DIR = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOK_DIR, local_files_only=True)

    print("Loading datasets...")
    # Assuming you have train.json and dev.json
    train_dataset = SQuADDataset('train-v2.0.json', tokenizer, max_len=MAX_LEN)
    val_dataset = SQuADDataset('dev-v2.0.json', tokenizer, max_len=MAX_LEN)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True  # Important for GPU performance
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print("Initializing model...")
    model = TransformerQA(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Save model
    torch.save(model.state_dict(), 'transformer_qa_model.pth')
    print("Model saved!")

if __name__ == "__main__":
    main()