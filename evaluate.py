#!/usr/bin/env python3
"""
Fast evaluator for SQuAD v2-style extractive QA models.

- Runs the model ONCE to collect, for each question:
    * best non-null span text and score (start+end logits)
    * null score (CLS-based or explicit null logit)
- Then sweeps null-thresholds WITHOUT doing more inference.

Usage example:
  python evaluate.py \
    --model outputs/transformer_qa_636008719.pth \
    --data dev-v2.0.json \
    --tok_dir ./bert-base-uncased \
    --window_batch_size 16 \
    --tune_threshold \
    --thr_start -15 --thr_end 15 --thr_step 0.5
"""

import argparse, json, math, string, re, sys, os
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- import your model ---
from transformer_qa import TransformerQA

# ----------------- utils -----------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):  return " ".join(text.split())
    def remove_punc(text):      return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):            return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(pred_tokens), 1)
    recall    = num_same / max(len(gt_tokens), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_predictions(predictions, data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    f1s, ems = [], []
    has_ids, no_ids = set(), set()

    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                qid = qa["id"]
                if qa.get("is_impossible", False):
                    no_ids.add(qid)
                else:
                    has_ids.add(qid)
                if qid not in predictions:
                    continue
                pred = predictions[qid]
                gts = [a["text"] for a in qa.get("answers", [])] or [""]
                f1s.append(max(f1_score(pred, gt) for gt in gts))
                ems.append(max(exact_match_score(pred, gt) for gt in gts))

    n_all = len(f1s)
    avg_f1 = 100 * (sum(f1s) / n_all if n_all else 0.0)
    avg_em = 100 * (sum(ems) / n_all if n_all else 0.0)

    # Split metrics for has/no-answer to help debugging
    has_f1 = []
    has_em = []
    no_em  = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                qid = qa["id"]
                if qid not in predictions:
                    continue
                pred = predictions[qid]
                gts  = [a["text"] for a in qa.get("answers", [])] or [""]
                if qa.get("is_impossible", False):
                    no_em.append(max(exact_match_score(pred, gt) for gt in gts))
                else:
                    has_f1.append(max(f1_score(pred, gt) for gt in gts))
                    has_em.append(max(exact_match_score(pred, gt) for gt in gts))

    def pct(x): return 100 * (sum(x) / len(x)) if x else 0.0
    split = {
        "n_all": n_all,
        "n_has": len(has_ids),
        "n_no":  len(no_ids),
        "has_f1": pct(has_f1),
        "has_em": pct(has_em),
        "no_em":  pct(no_em),
    }
    return avg_f1, avg_em, split

# ----------------- model loading -----------------
def load_model_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # our training script saved a dict with extra metadata
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        # plain state dict
        model.load_state_dict(ckpt, strict=False)
    else:
        raise ValueError("Unknown checkpoint format")
    return model

# ----------------- span search -----------------
def best_span_from_logits(start_log, end_log, token_type_ids, offsets, max_answer_len=30):
    """
    Returns (best_score, (start_idx, end_idx)) constrained to context tokens (token_type_id==1),
    ignoring special tokens with offset (0,0).
    """
    # find where context begins
    ctx_start = 0
    for i, t in enumerate(token_type_ids):
        if t == 1:
            ctx_start = i
            break

    L = len(start_log)
    best_score = -1e30
    best_s, best_e = 0, 0

    # Simply brute-force up to max_answer_len; L is small (<=384)
    for s in range(ctx_start, L):
        if offsets[s][1] == 0:  # special token
            continue
        # end cannot be before start, and cap by max length
        e_max = min(L - 1, s + max_answer_len - 1)
        for e in range(s, e_max + 1):
            if offsets[e][1] == 0:
                continue
            score = start_log[s] + end_log[e]
            if score > best_score:
                best_score, best_s, best_e = score, s, e
    return best_score, (best_s, best_e)

def tokens_to_text(tokenizer, input_ids, start_idx, end_idx):
    toks = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
    text = tokenizer.convert_tokens_to_string(toks).strip()
    # clean artifacts
    return text.replace("[CLS]", "").replace("[SEP]", "").strip()

# ----------------- candidate collection -----------------
@torch.inference_mode()
def collect_candidates(model, tokenizer, data_path, device, max_len=384, stride=128, window_batch_size=16, max_answer_len=30):
    """
    One full pass. For each QID, keep:
      - best non-null text and score
      - null score
    Returns: dict[qid] = {"best_text": str, "best_score": float, "null_score": float}
    """
    model.eval()
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = {}
    # Iterate by article for a stable progress-bar (35 for SQuAD dev)
    for article in tqdm(dataset["data"], desc="Evaluating", leave=False):
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                question = qa["question"]

                # Tokenize into sliding windows
                enc = tokenizer(
                    question,
                    context,
                    truncation="only_second",
                    max_length=max_len,
                    stride=stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    return_token_type_ids=True,
                    padding="max_length"
                )
                n = len(enc["input_ids"])
                best_text = ""
                best_score = -1e30
                null_score_global = -1e30

                # batched windows
                for i0 in range(0, n, window_batch_size):
                    i1 = min(n, i0 + window_batch_size)
                    batch = {k: torch.tensor(enc[k][i0:i1]).to(device) for k in ["input_ids","attention_mask","token_type_ids"]}
                    offsets_batch = enc["offset_mapping"][i0:i1]

                    out = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
                    if isinstance(out, tuple) and len(out) == 3:
                        start_logits, end_logits, null_logits = out
                        null_vec = null_logits  # shape [B]
                    else:
                        start_logits, end_logits = out
                        # CLS is index 0
                        null_vec = start_logits[:, 0] + end_logits[:, 0]

                    # Move to CPU numpy for easy loops
                    start_np = start_logits.detach().float().cpu().numpy()
                    end_np   = end_logits.detach().float().cpu().numpy()
                    null_np  = null_vec.detach().float().cpu().numpy()
                    input_ids_batch = batch["input_ids"].detach().cpu().tolist()
                    ttids_batch     = batch["token_type_ids"].detach().cpu().tolist()

                    for b in range(i1 - i0):
                        null_score_global = max(null_score_global, float(null_np[b]))
                        score, (s_idx, e_idx) = best_span_from_logits(
                            start_np[b], end_np[b], ttids_batch[b], offsets_batch[b], max_answer_len
                        )
                        if score > best_score:
                            best_score = float(score)
                            best_text  = tokens_to_text(tokenizer, input_ids_batch[b], s_idx, e_idx)

                # if no valid non-null found, keep empty with very low score
                results[qid] = {
                    "best_text": best_text,
                    "best_score": best_score,
                    "null_score": null_score_global
                }
    return results

# ----------------- threshold sweep (cheap) -----------------
def predict_with_threshold(candidates, threshold: float):
    """
    Apply HF-style rule once candidates are cached:
      empty if null_score > best_score + threshold  (SQuAD v2) 
    """
    preds = {}
    empty = 0
    lengths = []
    for qid, rec in candidates.items():
        if rec["null_score"] > (rec["best_score"] + threshold):
            preds[qid] = ""
            empty += 1
        else:
            preds[qid] = rec["best_text"]
            lengths.append(len(rec["best_text"].split()))
    empty_pct = 100.0 * empty / max(len(candidates), 1)
    avg_span = sum(lengths) / max(len(lengths), 1) if lengths else 0.0
    return preds, empty_pct, avg_span

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--data", type=str, default="dev-v2.0.json")
    ap.add_argument("--tok_dir", type=str, default="bert-base-uncased")
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--max_answer_length", type=int, default=30)
    ap.add_argument("--window_batch_size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.0, help="Used if --tune_threshold is not set")
    ap.add_argument("--tune_threshold", action="store_true", help="Grid-search threshold AFTER a single model pass")
    ap.add_argument("--thr_start", type=float, default=-15.0)
    ap.add_argument("--thr_end", type=float, default=15.0)
    ap.add_argument("--thr_step", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True)

    # Build model with typical defaults used in training (adapt if yours differ)
    model = TransformerQA(
        vocab_size=tokenizer.vocab_size,
        d_model=256, n_heads=8, n_layers=4, d_ff=1024, max_len=args.max_len, dropout=0.0
    ).to(device)
    load_model_state(model, args.model, device)
    model.eval()

    print("Collecting per-question candidates (one model pass)...")
    candidates = collect_candidates(
        model, tokenizer, args.data, device,
        max_len=args.max_len, stride=args.stride,
        window_batch_size=args.window_batch_size,
        max_answer_len=args.max_answer_length,
    )

    if args.tune_threshold:
        print(f"Tuning null threshold from {args.thr_start} to {args.thr_end} step {args.thr_step} ...")
        best = (-1.0, -1.0, None, None)  # (F1, EM, thr, dbg)
        thr = args.thr_start
        while thr <= args.thr_end + 1e-9:
            preds, empty_pct, avg_span = predict_with_threshold(candidates, thr)
            f1, em, split = evaluate_predictions(preds, args.data)
            if f1 > best[0]:
                best = (f1, em, thr, (empty_pct, avg_span, split))
            thr += args.thr_step

        f1, em, thr, (empty_pct, avg_span, split) = best
        print(f"[TUNED] best_null_threshold={thr:.2f}  F1={f1:.2f}%  EM={em:.2f}%")
        print(f"HasAns F1: {split['has_f1']:.2f}%   HasAns EM: {split['has_em']:.2f}%")
        print(f"NoAns EM:  {split['no_em']:.2f}%")
        print(f"[DBG] empty%={empty_pct:.2f}%  avg_span_tokens≈{avg_span:.1f}  "
              f"(n_all={split['n_all']}  n_has={split['n_has']}  n_no={split['n_no']})")

        # Save final predictions at tuned threshold
        preds, _, _ = predict_with_threshold(candidates, thr)
    else:
        preds, empty_pct, avg_span = predict_with_threshold(candidates, args.threshold)
        f1, em, split = evaluate_predictions(preds, args.data)
        print("\n========================================")
        print("EVALUATION RESULTS")
        print("========================================")
        print(f"F1 Score: {f1:.2f}%")
        print(f"Exact Match: {em:.2f}%")
        print(f"HasAns F1: {split['has_f1']:.2f}%   HasAns EM: {split['has_em']:.2f}%")
        print(f"NoAns EM:  {split['no_em']:.2f}%")
        print(f"[DBG] empty%={empty_pct:.2f}%  avg_span_tokens≈{avg_span:.1f}  "
              f"(n_all={split['n_all']}  n_has={split['n_has']}  n_no={split['n_no']})")
        print("========================================")

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print("Predictions saved to predictions.json")

if __name__ == "__main__":
    main()
