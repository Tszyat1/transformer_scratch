# transformer_qa.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        self._init_weights()

    def _init_weights(self):
        for m in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))

    def forward(self, query, key, value, mask=None):
        """
        query/key/value: (B, L, D)
        mask: (B, L) with 1 for keep, 0 for mask out (same convention as attention_mask)
        """
        bsz, seqlen, _ = query.size()

        # Project
        Q = self.w_q(query).view(bsz, seqlen, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,L,Dk)
        K = self.w_k(key  ).view(bsz, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(bsz, seqlen, self.n_heads, self.d_k).transpose(1, 2)

        # Scores (compute and softmax in fp32 for AMP stability)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale          # (B,H,L,L)
        scores_f32 = scores.float()
        if mask is not None:
            # broadcast mask to (B,1,1,L) ; 1=keep, 0=mask
            mask4d = mask.unsqueeze(1).unsqueeze(1)
            scores_f32 = scores_f32.masked_fill(mask4d == 0, -1e4)

        attn = torch.softmax(scores_f32, dim=-1).to(Q.dtype)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)  # (B,H,L,Dk)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.w_o(ctx)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        return self.lin2(self.dropout(self.gelu(self.lin1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x

class TransformerQA(nn.Module):
    """Transformer encoder (from scratch) + QA heads (start/end).
       Use [CLS] (index 0) as the null span for SQuAD v2."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 d_ff=1024, max_len=384, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)  # 0=question,1=context
        self.posenc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # Heads
        self.qa_outputs = nn.Linear(d_model, 2)  # start/end

        # inits
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.qa_outputs.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embedding(input_ids)
        if token_type_ids is not None:
            x = x + self.token_type_embedding(token_type_ids)
        x = x * math.sqrt(x.size(-1))

        x = x.transpose(0, 1)
        x = self.posenc(x)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, attention_mask)

        h = self.dropout(x)
        logits = self.qa_outputs(h)
        start_logits, end_logits = logits[..., 0], logits[..., 1]
        return start_logits, end_logits
