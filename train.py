# train.py
import argparse, json, math, os, random
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from transformer_qa import TransformerQA

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset
# -----------------------------
@dataclass
class Feature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    start_position: int
    end_position: int
    has_answer: bool

class SQuADDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=384, stride=128, neg_ratio=1.0, is_train=True):
        self.tok = tokenizer
        self.max_len = max_len
        self.stride = stride
        self.is_train = is_train
        self.neg_ratio = neg_ratio
        self.features: List[Feature] = []

        # debug counters
        self.debug_total_windows = 0
        self.debug_missed_spans = 0
        self.debug_impossible_total = 0
        self.debug_impossible_kept = 0

        with open(data_path, "r") as f:
            data = json.load(f)["data"]

        for article in tqdm(data, desc="Processing articles"):
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    q = qa["question"]; qid = qa["id"]
                    has_ans = not qa.get("is_impossible", False) and len(qa.get("answers", [])) > 0
                    if has_ans:
                        ans = qa["answers"][0]
                        a_start = ans["answer_start"]
                        a_end = a_start + len(ans["text"])
                        self._add_windows(q, context, a_start, a_end, True)
                    else:
                        self.debug_impossible_total += 1
                        self._add_windows(q, context, -1, -1, False)

        # Negative downsample (train only)
        if self.is_train and self.neg_ratio < 1.0:
            keep = []
            neg_kept = 0
            neg_total = 0
            for f in self.features:
                if not f.has_answer:
                    neg_total += 1
                    if random.random() < self.neg_ratio:
                        keep.append(f); neg_kept += 1
                else:
                    keep.append(f)
            self.features = keep
            self.debug_impossible_kept = neg_kept
        else:
            self.debug_impossible_kept = sum(1 for f in self.features if not f.has_answer)

        print(f"Created {len(self.features)} examples")
        print(f"[DEBUG] windows={self.debug_total_windows}  missed_spans={self.debug_missed_spans}  "
              f"impossible_total={self.debug_impossible_total}  impossible_kept={self.debug_impossible_kept}")

    def _add_windows(self, question, context, a_start, a_end, has_answer):
        enc = self.tok(
            question, context,
            truncation="only_second",
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True
        )
        n = len(enc["input_ids"])
        found_any = False
        for i in range(n):
            input_ids = enc["input_ids"][i]
            attn      = enc["attention_mask"][i]
            ttids     = enc["token_type_ids"][i]
            offsets   = enc["offset_mapping"][i]

            # context begins where token_type_ids == 1
            ctx_start = next((k for k,t in enumerate(ttids) if t == 1), 0)

            start_pos = end_pos = 0  # [CLS]
            ok = True
            if has_answer:
                # find token indices overlapping the gold char span
                found_s = found_e = False
                for idx in range(ctx_start, len(offsets)):
                    s, e = offsets[idx]
                    if s == e:   # special tokens
                        continue
                    if not found_s and s <= a_start < e:
                        start_pos = idx; found_s = True
                    if s < a_end <= e:
                        end_pos = idx; found_e = True
                        break
                ok = found_s and found_e
                found_any |= ok

            if not ok:
                self.debug_missed_spans += 1
                continue

            self.features.append(Feature(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attn, dtype=torch.long),
                token_type_ids=torch.tensor(ttids, dtype=torch.long),
                start_position=start_pos,
                end_position=end_pos,
                has_answer=has_answer
            ))
            self.debug_total_windows += 1

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        f = self.features[idx]
        return {
            "input_ids": f.input_ids,
            "attention_mask": f.attention_mask,
            "token_type_ids": f.token_type_ids,
            "start_position": torch.tensor(f.start_position, dtype=torch.long),
            "end_position": torch.tensor(f.end_position, dtype=torch.long),
            "has_answer": torch.tensor(1 if f.has_answer else 0, dtype=torch.long)
        }

# -----------------------------
# Training / Eval
# -----------------------------
def mask_non_context_logits(start_logits, end_logits, token_type_ids):
    """
    Mask everything except context tokens (ttid==1) and [CLS] at index 0.
    Works in fp16/fp32.
    """
    B, L = start_logits.size()
    allow = (token_type_ids == 1)
    allow[:, 0] = True  # allow [CLS]
    very_neg = torch.tensor(-1e4, dtype=start_logits.dtype, device=start_logits.device)
    start_logits = torch.where(allow, start_logits, very_neg)
    end_logits   = torch.where(allow, end_logits, very_neg)
    return start_logits, end_logits

def train_model(model, train_loader, val_loader, args):
    model.to(DEVICE)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and DEVICE.type == "cuda")

    # Optim + sched
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)
    total_steps = len(train_loader) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction="none")

    print("\n==================================================")
    print(f"TRAINING ON: {DEVICE}")
    print("==================================================\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attn      = batch["attention_mask"].to(DEVICE, non_blocking=True)
            ttids     = batch["token_type_ids"].to(DEVICE, non_blocking=True)
            y_s       = batch["start_position"].to(DEVICE, non_blocking=True)
            y_e       = batch["end_position"].to(DEVICE, non_blocking=True)
            has_ans   = batch["has_answer"].to(DEVICE, non_blocking=True)  # 1/0

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and DEVICE.type == "cuda"):
                s_logits, e_logits = model(input_ids, attn, ttids)

                if args.restrict_ctx_loss:
                    s_logits, e_logits = mask_non_context_logits(s_logits, e_logits, ttids)

                # per-sample CE
                s_loss = ce(s_logits, y_s)
                e_loss = ce(e_logits, y_e)
                span_loss = 0.5 * (s_loss + e_loss)

                # weight positives vs negatives
                w = torch.where(has_ans == 1,
                                torch.full_like(span_loss, args.pos_span_weight),
                                torch.full_like(span_loss, args.neg_span_weight))
                loss = (span_loss * w).sum() / (w.sum() + 1e-8)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running / len(train_loader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # quick val
        if val_loader is not None:
            model.eval(); val_running = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    input_ids = batch["input_ids"].to(DEVICE)
                    attn      = batch["attention_mask"].to(DEVICE)
                    ttids     = batch["token_type_ids"].to(DEVICE)
                    y_s       = batch["start_position"].to(DEVICE)
                    y_e       = batch["end_position"].to(DEVICE)

                    s_logits, e_logits = model(input_ids, attn, ttids)
                    if args.restrict_ctx_loss:
                        s_logits, e_logits = mask_non_context_logits(s_logits, e_logits, ttids)
                    s_loss = ce(s_logits, y_s); e_loss = ce(e_logits, y_e)
                    loss = 0.5 * (s_loss.mean() + e_loss.mean())
                    val_running += loss.item()
            print(f"Validation Loss: {val_running / len(val_loader):.4f}")

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", type=str, required=True)
    ap.add_argument("--train_file", type=str, default="train-v2.0.json")
    ap.add_argument("--dev_file", type=str,   default="dev-v2.0.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="keep fraction of negatives")
    ap.add_argument("--pos_span_weight", type=float, default=1.0)
    ap.add_argument("--neg_span_weight", type=float, default=0.2)
    ap.add_argument("--restrict_ctx_loss", action="store_true")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(1337)

    print("Initializing tokenizer (local only)...")
    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True)
    print(f"Vocabulary size: {tok.vocab_size}\n")

    print("Loading datasets...")
    train_ds = SQuADDataset(args.train_file, tok, max_len=args.max_len, stride=args.stride,
                            neg_ratio=args.neg_ratio, is_train=True)
    dev_ds   = SQuADDataset(args.dev_file, tok, max_len=args.max_len, stride=args.stride,
                            neg_ratio=1.0, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate_fn)

    print("\nInitializing model...")
    model = TransformerQA(vocab_size=tok.vocab_size, d_model=256, n_heads=8,
                          n_layers=4, d_ff=1024, max_len=args.max_len, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    train_model(model, train_loader, val_loader, args)

    os.makedirs("outputs", exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024, "max_len": args.max_len},
        "vocab_size": tok.vocab_size
    }
    out = os.path.join("outputs", f"transformer_qa_{torch.randint(0, 10**9, ()).item()}.pth")
    torch.save(ckpt, out)
    print(f"\n✓ Model saved to: {out}")

if __name__ == "__main__":
    main()
