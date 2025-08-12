# train.py
import os, time, json, math, random
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from transformer_qa import TransformerQA

# --------------------------
# Repro & device utilities
# --------------------------
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU AVAILABLE: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Current GPU: {torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        print("✗ WARNING: GPU NOT AVAILABLE - training will be slow")
    return device

DEVICE = check_gpu()
set_seed(1337)

# --------------------------
# SQuAD dataset (v1 & v2)
# --------------------------
class SQuADDataset(Dataset):
    """
    Builds training/eval windows using sliding window tokenization.
    - Robust char->token mapping via offsets (not strict equality).
    - SQuAD v2 negatives: label = [CLS] (index 0,0).
    - Downsample unanswerable QAs with --neg_ratio, keep ONE negative window per kept QA.
      (Reduces null bias for from-scratch models.)
    """
    def __init__(self, data_path: str, tokenizer, max_len=384, stride=128, neg_ratio=1.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.neg_ratio = float(neg_ratio)
        self.features: List[Dict] = []
        self.debug_total = 0
        self.debug_impossible_total = 0
        self.debug_impossible_kept = 0
        self.debug_missed = 0

        print(f"Loading dataset from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        self._preprocess(js["data"])
        print(f"Created {len(self.features)} examples")
        print(f"[DEBUG] windows={self.debug_total}  "
              f"missed_spans={self.debug_missed}  "
              f"impossible_total={self.debug_impossible_total}  "
              f"impossible_kept={self.debug_impossible_kept}")

    def _preprocess(self, data):
        for article in tqdm(data, desc="Processing articles"):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qid = qa["id"]
                    question = qa["question"]
                    has_ans = (not qa.get("is_impossible", False)) and qa.get("answers")
                    if has_ans:
                        ans = qa["answers"][0]
                        a_start = ans["answer_start"]
                        a_end   = a_start + len(ans["text"])
                        self._create_features(question, context, a_start, a_end, qid, has_answer=True)
                    else:
                        # downsample unanswerable QAs
                        self.debug_impossible_total += 1
                        keep = (random.random() < self.neg_ratio)
                        if keep:
                            self._create_features(question, context, -1, -1, qid, has_answer=False)
                            self.debug_impossible_kept += 1

    def _create_features(self, question, context, answer_start, answer_end, qid, has_answer=True):
        enc = self.tokenizer(
            question, context,
            truncation="only_second",
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True,
        )

        neg_kept = False  # keep only one window for negatives
        for i in range(len(enc["input_ids"])):
            input_ids = enc["input_ids"][i]
            attn_mask = enc["attention_mask"][i]
            ttids     = enc["token_type_ids"][i]
            offsets   = enc["offset_mapping"][i]

            # context begins where token_type_ids == 1
            context_start = next((k for k,t in enumerate(ttids) if t == 1), 0)

            start_pos = end_pos = 0  # CLS by default (no-answer)
            found = False

            if has_answer:
                # interval overlap mapping, skip special tokens with (0,0) offsets
                for idx in range(context_start, len(offsets)):
                    s, e = offsets[idx]
                    if s == e:  # special token (e.g., [CLS], [SEP])
                        continue
                    if (s <= answer_start < e) and not found:
                        start_pos = idx
                        found = True
                    if (s < answer_end <= e):
                        end_pos = idx
                        if found:
                            break
                if not found:
                    self.debug_missed += 1
                    continue
            else:
                # keep only a single negative window per unanswerable QA
                if neg_kept:
                    continue
                neg_kept = True
                start_pos = end_pos = 0

            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(ttids, dtype=torch.long),
                "start_position": torch.tensor(start_pos, dtype=torch.long),
                "end_position": torch.tensor(end_pos, dtype=torch.long),
                "qid": qid
            })
            self.debug_total += 1

    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx]

# --------------------------
# Train / Eval (loss) loops
# --------------------------
def train_model(model, train_loader, val_loader, epochs=3, lr=1e-4, tokenizer=None):
    print("\n" + "="*50)
    print(f"TRAINING ON: {DEVICE}")
    print("="*50 + "\n")

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            ttids = batch["token_type_ids"].to(DEVICE)
            start_pos = batch["start_position"].to(DEVICE)
            end_pos = batch["end_position"].to(DEVICE)

            if step % 200 == 0:
                assert input_ids.is_cuda, "ERROR: tensors not on GPU!"

            start_logits, end_logits = model(input_ids, attn, ttids)
            loss = (criterion(start_logits, start_pos) + criterion(end_logits, end_pos)) / 2.0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        if val_loader:
            v = evaluate_loss(model, val_loader, criterion)
            print(f"Validation Loss: {v:.4f}")

def evaluate_loss(model, data_loader, criterion):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            ttids = batch["token_type_ids"].to(DEVICE)
            start_pos = batch["start_position"].to(DEVICE)
            end_pos = batch["end_position"].to(DEVICE)

            sl, el = model(input_ids, attn, ttids)
            loss = (criterion(sl, start_pos) + criterion(el, end_pos)) / 2.0
            total += loss.item()
    return total / len(data_loader)

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train-v2.0.json")
    parser.add_argument("--dev_file",   type=str, default="dev-v2.0.json")
    parser.add_argument("--tok_dir",    type=str, default="./bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=6)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max_len",    type=int, default=384)
    parser.add_argument("--stride",     type=int, default=128)
    parser.add_argument("--neg_ratio",  type=float, default=0.33,
                        help="Keep this fraction of unanswerable QAs (one window each). 1.0=keep all.")
    args = parser.parse_args()

    print("Initializing tokenizer (local only)...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("\nLoading datasets...")
    train_dataset = SQuADDataset(args.train_file, tokenizer,
                                 max_len=args.max_len, stride=args.stride, neg_ratio=args.neg_ratio)
    val_dataset   = SQuADDataset(args.dev_file,   tokenizer,
                                 max_len=args.max_len, stride=args.stride, neg_ratio=1.0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print("\nInitializing model...")
    model = TransformerQA(
        vocab_size=tokenizer.vocab_size,
        d_model=256, n_heads=8, n_layers=4, d_ff=1024,
        max_len=args.max_len, dropout=0.1
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, tokenizer=tokenizer)

    # Save a richer checkpoint so evaluate.py can reconstruct the model
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024,
                   "max_len": args.max_len, "dropout": 0.0},
        "vocab_size": tokenizer.vocab_size,
    }
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/transformer_qa_{time.strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(ckpt, out)
    print(f"\n✓ Model saved to: {out}")
