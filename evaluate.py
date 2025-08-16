#!/usr/bin/env python3
import argparse, json, math, os, sys, re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_text(s: str) -> str:
    """SQuAD-style text normalization for EM/F1."""
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return re.sub(r"[^\w\s]", "", text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)

def make_attention_mask(token_type_ids: torch.Tensor) -> torch.Tensor:
    """Build full attention mask (1 for real tokens) from token_type_ids != -100 or attention_mask given."""
    # token_type_ids here is fine; actual attention_mask comes from dataset.
    return None

# ---------------------------------------------------------------------
# Data: minimal SQuADv2 windowed dataset with offsets
# ---------------------------------------------------------------------

class SQuADWindowDataset(Dataset):
    """
    Creates overlapping windows per (question, paragraph) with offset mapping
    so we can map token spans back to character substrings.
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizerFast,
                 max_len: int = 384,
                 stride: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stride = stride

        with open(data_path, "r", encoding="utf-8") as f:
            js = json.load(f)

        self.examples = []  # list of dicts, each is one window
        for article in tqdm(js["data"], desc="Building windows"):
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    qid = qa["id"]
                    question = qa["question"]
                    answers = [a["text"] for a in qa.get("answers", [])]
                    # impossible = qa.get("is_impossible", False)  # not strictly needed for eval

                    enc = self.tokenizer(
                        question,
                        context,
                        truncation="only_second",
                        max_length=self.max_len,
                        stride=self.stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        padding="max_length",
                        return_token_type_ids=True,
                    )

                    for i in range(len(enc["input_ids"])):
                        input_ids = enc["input_ids"][i]
                        attention_mask = enc["attention_mask"][i]
                        token_type_ids = enc["token_type_ids"][i]
                        offsets = enc["offset_mapping"][i]  # list of (char_start, char_end)
                        self.examples.append({
                            "qid": qid,
                            "question": question,
                            "context": context,
                            "answers": answers,
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "offsets": offsets,
                        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return (
            ex["qid"],
            torch.tensor(ex["input_ids"], dtype=torch.long),
            torch.tensor(ex["attention_mask"], dtype=torch.long),
            torch.tensor(ex["token_type_ids"], dtype=torch.long),
            ex["offsets"],
            ex["context"],
            ex["answers"],
        )

# ---------------------------------------------------------------------
# Model loader: rebuild architecture from checkpoint config
# ---------------------------------------------------------------------

def build_model_from_ckpt(ckpt_path: str, vocab_size: int):
    """
    Reads ckpt['config'] to rebuild TransformerQA with matching dims,
    then loads weights. Works with checkpoints saved by your training scripts.
    """
    from transformer_qa import TransformerQA  # your local module

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Accept both full dict and raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
    else:
        state_dict = ckpt
        cfg = {}

    d_model = int(cfg.get("d_model", 256))
    n_heads = int(cfg.get("n_heads", 8))
    n_layers = int(cfg.get("n_layers", 4))
    d_ff = int(cfg.get("d_ff", 1024))
    dropout = float(cfg.get("dropout", 0.1))
    max_len = int(cfg.get("max_len", 384))
    # print config for clarity
    print(f"[MODEL CONFIG] d_model={d_model}  n_heads={n_heads}  n_layers={n_layers}  d_ff={d_ff}  dropout={dropout}  max_len={max_len}")

    model = TransformerQA(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) < 50:
            print(" missing:", missing)
            print(" unexpected:", unexpected)
    model.to(DEVICE)
    model.eval()
    return model, max_len

# ---------------------------------------------------------------------
# Scoring / span extraction
# ---------------------------------------------------------------------

def get_best_span_from_window(start_logits: torch.Tensor,
                              end_logits: torch.Tensor,
                              token_type_ids: torch.Tensor,
                              offsets: List[Tuple[int, int]],
                              max_answer_len: int = 30) -> Tuple[float, Tuple[int,int]]:
    """
    Returns best span score and (start_idx, end_idx) *token indices* within this window.
    Restrict to context tokens where token_type_ids == 1.
    """
    # Mask to context tokens
    tt = token_type_ids.bool()
    s = start_logits.clone()
    e = end_logits.clone()
    s[~tt] = -1e9
    e[~tt] = -1e9

    # Outer sum to get span scores (start<=end and length<=max_answer_len)
    L = s.size(0)
    span_scores = s[:, None] + e[None, :]
    tri_mask = torch.triu(torch.ones((L, L), device=span_scores.device), diagonal=0)
    len_mask = torch.tril(torch.ones((L, L), device=span_scores.device), diagonal=max_answer_len-1)
    valid = (tri_mask * len_mask).bool()
    span_scores[~valid] = -1e9

    # Also block CLS token (index 0) as a span token
    span_scores[0, :] = -1e9
    span_scores[:, 0] = -1e9

    flat_idx = torch.argmax(span_scores).item()
    i = flat_idx // L
    j = flat_idx % L
    best_score = span_scores[i, j].item()
    return best_score, (i, j)

def span_to_text(context: str, offsets: List[Tuple[int,int]], start_idx: int, end_idx: int) -> str:
    char_start = offsets[start_idx][0]
    char_end   = offsets[end_idx][1]
    if char_start is None or char_end is None:
        return ""
    char_start = max(0, char_start)
    char_end = max(char_start, char_end)
    return context[char_start:char_end]

# ---------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------

def collect_candidates(model,
                       data_loader: DataLoader,
                       max_answer_len: int = 30,
                       window_batch_size: int = 16):
    """
    One forward pass over all windows; aggregate best span per question (and null scores).
    Returns:
      per_q_best: qid -> (best_span_score, best_text)
      per_q_null: qid -> best_null_score  (we use start[CLS]+end[CLS] as null score)
      golds     : qid -> List[str] (gold answers)
    """
    per_q_best: Dict[str, Tuple[float, str]] = {}
    per_q_null: Dict[str, float] = defaultdict(lambda: -1e9)
    golds: Dict[str, List[str]] = {}

    with torch.no_grad():
        buf = []
        meta = []

        def flush():
            nonlocal buf, meta
            if not buf: return
            input_ids = torch.stack([b[0] for b in buf]).to(DEVICE)
            attn      = torch.stack([b[1] for b in buf]).to(DEVICE)
            ttids     = torch.stack([b[2] for b in buf]).to(DEVICE)

            start_logits, end_logits = model(input_ids, attn, ttids)
            start_logits = start_logits.float().cpu()
            end_logits   = end_logits.float().cpu()
            ttids_cpu    = ttids.cpu()

            for k in range(start_logits.size(0)):
                qid, offsets, context, answers = meta[k]
                # null score as CLS start+end
                null_score = (start_logits[k, 0] + end_logits[k, 0]).item()
                if null_score > per_q_null[qid]:
                    per_q_null[qid] = null_score
                # best span within window
                score, (si, ei) = get_best_span_from_window(
                    start_logits[k], end_logits[k], ttids_cpu[k], offsets, max_answer_len
                )
                text = span_to_text(context, offsets, si, ei)
                prev = per_q_best.get(qid, (-1e9, ""))
                if score > prev[0]:
                    per_q_best[qid] = (score, text)
                if qid not in golds:
                    golds[qid] = answers

            buf = []
            meta = []

        for batch in tqdm(data_loader, desc="Collecting windows"):
            # each batch item: (qid, input_ids, attention_mask, token_type_ids, offsets, context, answers)
            for i in range(len(batch[0])):
                qid = batch[0][i]
                input_ids = batch[1][i]
                attn = batch[2][i]
                ttids = batch[3][i]
                offsets = batch[4][i]
                context = batch[5][i]
                answers = batch[6][i]
                buf.append((input_ids, attn, ttids))
                meta.append((qid, offsets, context, answers))
                if len(buf) == window_batch_size:
                    flush()
        flush()
    return per_q_best, per_q_null, golds

def evaluate_grid(per_q_best, per_q_null, golds,
                  thr_start: float, thr_end: float, thr_step: float) -> Tuple[float, float, float, float, float]:
    """
    Try thresholds on (best_span_score - null_score). If < threshold, predict empty.
    Returns best (F1, EM, has_f1, has_em, no_em) and prints the tuned threshold.
    """
    qids = list(golds.keys())
    best = (-1.0, -1.0, -1.0, -1.0, -1.0, 0.0)

    thr = thr_start
    while thr <= thr_end + 1e-9:
        preds = {}
        for qid in qids:
            span_score, text = per_q_best.get(qid, (-1e9, ""))
            null_score = per_q_null.get(qid, -1e9)
            margin = span_score - null_score
            if margin >= thr and text.strip():
                preds[qid] = text
            else:
                preds[qid] = ""  # predict null
        f1, em, has_f1, has_em, no_em = compute_metrics(preds, golds)
        if f1 > best[0]:
            best = (f1, em, has_f1, has_em, no_em, thr)
        thr += thr_step
    print(f"[TUNED] best_null_threshold={best[5]:.2f}  F1={best[0]*100:.2f}%  EM={best[1]*100:.2f}%")
    return best[:5]

def compute_metrics(preds: Dict[str,str], golds: Dict[str, List[str]]):
    n_all = len(golds)
    n_has = 0
    n_no  = 0
    f1_sum = 0.0
    em_sum = 0.0
    has_f1 = has_em = 0.0
    no_em  = 0.0

    for qid, answers in golds.items():
        is_has = len(answers) > 0 and any(a.strip() for a in answers)
        if is_has: n_has += 1
        else: n_no += 1
        pred = preds.get(qid, "")
        # pick max over references
        f1 = max((f1_score(pred, a) for a in answers), default=1.0 if pred=="" else 0.0)
        em = max((1.0 if exact_match_score(pred, a) else 0.0 for a in answers), default=1.0 if pred=="" else 0.0)
        f1_sum += f1
        em_sum += em
        if is_has:
            has_f1 += f1
            has_em += em
        else:
            no_em += 1.0 if pred == "" else 0.0

    f1 = f1_sum / max(1, n_all)
    em = em_sum / max(1, n_all)
    has_f1 = has_f1 / max(1, n_has)
    has_em = has_em / max(1, n_has)
    no_em = no_em / max(1, n_no)
    return f1, em, has_f1, has_em, no_em

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--data", required=True, help="SQuAD v2.0 dev file (json)")
    ap.add_argument("--tok_dir", required=True, help="Tokenizer dir (e.g., ./bert-base-uncased)")
    ap.add_argument("--window_batch_size", type=int, default=16)
    ap.add_argument("--max_answer_length", type=int, default=30)
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--thr_start", type=float, default=-2.0)
    ap.add_argument("--thr_end", type=float, default=8.0)
    ap.add_argument("--thr_step", type=float, default=0.25)
    ap.add_argument("--max_len", type=int, default=None, help="Override max_len for windowing (else use ckpt config or 384)")
    ap.add_argument("--stride", type=int, default=128)
    return ap.parse_args()

def main():
    args = parse_args()
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_dir, use_fast=True, local_files_only=True)
    model, cfg_max_len = build_model_from_ckpt(args.model, tokenizer.vocab_size)

    max_len = args.max_len if args.max_len is not None else cfg_max_len
    print(f"Using max_len={max_len} stride={args.stride}")

    print("Preparing evaluation dataset...")
    ds = SQuADWindowDataset(args.data, tokenizer, max_len=max_len, stride=args.stride)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=lambda x: list(zip(*x)))

    print("Collecting per-question candidates (one model pass)...")
    per_q_best, per_q_null, golds = collect_candidates(
        model, dl, max_answer_len=args.max_answer_length, window_batch_size=args.window_batch_size
    )

    if args.tune_threshold:
        print(f"Tuning null threshold from {args.thr_start} to {args.thr_end} step {args.thr_step} ...")
        f1, em, has_f1, has_em, no_em = evaluate_grid(per_q_best, per_q_null, golds, args.thr_start, args.thr_end, args.thr_step)
    else:
        # default thr=0
        preds = {}
        for qid, (span_score, text) in per_q_best.items():
            null_score = per_q_null.get(qid, -1e9)
            preds[qid] = text if (span_score - null_score) >= 0.0 and text.strip() else ""
        f1, em, has_f1, has_em, no_em = compute_metrics(preds, golds)

    print("\n========================================")
    print("EVALUATION RESULTS")
    print("========================================")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"Exact Match: {em*100:.2f}%")
    print(f"HasAns F1: {has_f1*100:.2f}%   HasAns EM: {has_em*100:.2f}%")
    print(f"NoAns EM:  {no_em*100:.2f}%")
    print("========================================")

    # Save a predictions.json for inspection (use best-threshold if tuned was run)
    preds = {}
    # choose threshold that was tuned if requested (best already printed); here we default to 0
    thr = 0.0
    for qid, (span_score, text) in per_q_best.items():
        null_score = per_q_null.get(qid, -1e9)
        preds[qid] = text if (span_score - null_score) >= thr and text.strip() else ""
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print("Predictions saved to predictions.json")

if __name__ == "__main__":
    main()
