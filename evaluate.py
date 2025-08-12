# evaluate.py
"""
Evaluation for SQuAD v1.1/v2.0-style extractive QA.
- windowed decoding (doc stride) with small batching for speed
- v2 "no-answer" via tuned threshold on (null_score - best_span_score)
- F1/EM with standard normalization
"""
import os, json, argparse, string, re
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_qa import TransformerQA

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU AVAILABLE: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("✗ GPU not available, using CPU")
    return device

DEVICE = check_gpu()

# -------- SQuAD scoring helpers (official-style normalization) --------
# HF docs mirror this normalization for QA. :contentReference[oaicite:1]{index=1}
def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens  = normalize_answer(prediction).split()
    gold_tokens  = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / max(1, len(pred_tokens))
    recall    = num_same / max(1, len(gold_tokens))
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_predictions(predictions, data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    f1s, ems = [], []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if qid not in predictions:
                    continue
                pred = predictions[qid]
                if qa.get("answers"):
                    gts = [a["text"] for a in qa["answers"]]
                else:
                    gts = [""]  # v2 no-answer ground truth
                f1 = max(f1_score(pred, gt) for gt in gts)
                em = max(exact_match_score(pred, gt) for gt in gts)
                f1s.append(f1); ems.append(em)

    avg_f1 = 100 * (sum(f1s) / max(1, len(f1s)))
    avg_em = 100 * (sum(ems) / max(1, len(ems)))
    return avg_f1, avg_em

# -------- Span search over windows + null handling --------
# Mirrors HF examples: compare best span vs null with a tunable threshold. :contentReference[oaicite:2]{index=2}
def best_span_per_item(start_logits, end_logits, token_type_ids, max_len, topk=50):
    # Restrict to context tokens only
    ctx = (token_type_ids == 1)
    s = start_logits.clone(); s[~ctx] = -1e9
    e = end_logits.clone();   e[~ctx] = -1e9

    k = min(len(s), topk)
    s_idx = torch.topk(s, k).indices
    e_idx = torch.topk(e, k).indices

    best = (-1e9, 0, 0)
    for si in s_idx:
        for ei in e_idx:
            si, ei = int(si), int(ei)
            if ei < si or (ei - si + 1) > max_len:
                continue
            score = s[si].item() + e[ei].item()
            if score > best[0]:
                best = (score, si, ei)
    return best  # (score, sidx, eidx)

def get_per_question_candidates(model, data_path, tokenizer,
                                max_len=384, stride=128, window_batch_size=16, max_answer_length=30):
    """
    For each question, compute:
      - best_non_null_score and its span text (across windows)
      - best null_score (CLS) across windows
    Returns dict: qid -> (diff=null-best, text=best_span_text)
    """
    model.eval()
    per_q = {}

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with torch.no_grad():
        for article in tqdm(data["data"], desc="Evaluating"):
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    qid = qa["id"]; question = qa["question"]

                    enc = tokenizer(
                        question, context,
                        truncation="only_second",
                        max_length=max_len,
                        stride=stride,
                        return_overflowing_tokens=True,
                        padding="max_length",
                        return_tensors="pt",
                        return_token_type_ids=True
                    )

                    # process windows in small batches for speed
                    W = enc["input_ids"].size(0)
                    best_overall = (-1e9, None, None, None)  # (score, s, e, tokens)
                    null_overall = -1e9

                    for start in range(0, W, window_batch_size):
                        end = min(W, start + window_batch_size)
                        input_ids = enc["input_ids"][start:end].to(DEVICE)
                        attn      = enc["attention_mask"][start:end].to(DEVICE)
                        ttids     = enc["token_type_ids"][start:end].to(DEVICE)

                        start_logits, end_logits = model(input_ids, attn, ttids)

                        for i in range(end - start):
                            s_logits = start_logits[i]; e_logits = end_logits[i]
                            tt = ttids[i].cpu()
                            ids = input_ids[i].cpu().tolist()

                            # null score at CLS (index 0)
                            null_score = (s_logits[0] + e_logits[0]).item()
                            if null_score > null_overall:
                                null_overall = null_score

                            score, sidx, eidx = best_span_per_item(s_logits, e_logits, tt, max_answer_length)
                            if score > best_overall[0]:
                                toks = tokenizer.convert_ids_to_tokens(ids)
                                best_overall = (score, sidx, eidx, toks)

                    # compute per-question diff and best span text
                    if best_overall[1] is None:
                        span_text = ""
                        best_score = -1e9
                    else:
                        sidx, eidx, toks = best_overall[1], best_overall[2], best_overall[3]
                        span_text = tokenizer.convert_tokens_to_string(toks[sidx:eidx+1]).strip()
                        span_text = span_text.replace("[CLS]", "").replace("[SEP]", "").strip()
                        best_score = best_overall[0]

                    diff = null_overall - best_score
                    per_q[qid] = (diff, span_text)
    return per_q

def apply_threshold(per_q, threshold):
    return {qid: ("" if diff > threshold else text) for qid, (diff, text) in per_q.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--data",  type=str, default="dev-v2.0.json")
    parser.add_argument("--tok_dir", type=str, default="./bert-base-uncased",
                        help="Local tokenizer dir (offline).")
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--stride",  type=int, default=128)
    parser.add_argument("--max_answer_length", type=int, default=30)
    parser.add_argument("--window_batch_size", type=int, default=16,
                        help="Number of windows per forward pass for speed.")
    parser.add_argument("--tune_threshold", action="store_true",
                        help="Sweep null threshold on dev to maximize F1.")
    parser.add_argument("--tune_start", type=float, default=-10.0)
    parser.add_argument("--tune_end",   type=float, default=20.0)
    parser.add_argument("--tune_step",  type=float, default=0.5)
    parser.add_argument("--use_threshold", type=float, default=0.0,
                        help="If not tuning, use this fixed threshold.")
    args = parser.parse_args()

    print("Loading model...")
    ckpt = torch.load(args.model, map_location=DEVICE)
    cfg = ckpt["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True)

    model = TransformerQA(
        vocab_size=ckpt["vocab_size"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg.get("dropout", 0.0)
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Getting per-question candidates (batched windows)...")
    per_q = get_per_question_candidates(
        model, args.data, tokenizer,
        max_len=args.max_len, stride=args.stride,
        window_batch_size=args.window_batch_size,
        max_answer_length=args.max_answer_length
    )

    # tune or use fixed threshold
    if args.tune_threshold:
        best = (-1.0, None, None)  # (F1, threshold, EM)
        T = args.tune_start
        thresholds = []
        while T <= args.tune_end + 1e-9:
            thresholds.append(round(T, 4))
            T += args.tune_step

        print(f"Tuning null threshold over {len(thresholds)} values...")
        for th in thresholds:
            preds = apply_threshold(per_q, th)
            f1, em = evaluate_predictions(preds, args.data)
            if f1 > best[0]:
                best = (f1, th, em)
        tuned_T = best[1]
        print(f"[TUNED] best_null_threshold={tuned_T:.2f}  F1={best[0]:.2f}%  EM={best[2]:.2f}%")
        final_preds = apply_threshold(per_q, tuned_T)
    else:
        final_preds = apply_threshold(per_q, args.use_threshold)

    print("Calculating scores...")
    f1, em = evaluate_predictions(final_preds, args.data)
    empty = sum(1 for v in final_preds.values() if v == "")
    avg_len = sum(len(p.split()) for p in final_preds.values()) / max(1, len(final_preds))

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"F1 Score: {f1:.2f}%")
    print(f"Exact Match: {em:.2f}%")
    print(f"[DBG] empty%={empty/len(final_preds):.2%}  avg_span_tokens≈{avg_len:.1f}")
    print("="*40)

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_preds, f, indent=2, ensure_ascii=False)
    print("Predictions saved to predictions.json")
