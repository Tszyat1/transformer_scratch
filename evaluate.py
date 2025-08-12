"""
Evaluation script for SQuAD Transformer
Calculates F1 and Exact Match scores
Usage: python evaluate.py --model outputs/transformer_qa_[timestamp].pth
"""

import torch
import json
import argparse
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_qa import TransformerQA, DEVICE
import string
import re

def normalize_answer(s):
    """Normalize answer for evaluation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def get_predictions(model, data_path, tokenizer, max_len=384):
    """Get predictions from model"""
    model.eval()
    predictions = {}
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with torch.no_grad():
        for article in tqdm(data['data'], desc="Evaluating"):
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    qid = qa['id']
                    question = qa['question']
                    
                    # Tokenize
                    inputs = tokenizer(
                        question,
                        context,
                        max_length=max_len,
                        truncation='only_second',
                        padding='max_length',
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    input_ids = inputs['input_ids'].to(DEVICE)
                    attention_mask = inputs['attention_mask'].to(DEVICE)
                    token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(DEVICE)
                    
                    # Get predictions
                    start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
                    
                    # Get most likely start and end
                    start_idx = torch.argmax(start_logits, dim=1).item()
                    end_idx = torch.argmax(end_logits, dim=1).item()
                    
                    # Ensure end >= start
                    if end_idx < start_idx:
                        end_idx = start_idx
                    
                    # Convert token indices to text
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                    answer_tokens = tokens[start_idx:end_idx+1]
                    answer = tokenizer.convert_tokens_to_string(answer_tokens)
                    
                    # Clean answer
                    answer = answer.replace('[CLS]', '').replace('[SEP]', '').strip()
                    
                    predictions[qid] = answer
    
    return predictions

def evaluate_predictions(predictions, data_path):
    """Calculate F1 and EM scores"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    f1_scores = []
    em_scores = []
    
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qid = qa['id']
                
                if qid not in predictions:
                    continue
                
                prediction = predictions[qid]
                
                # Get ground truth answers
                if qa.get('answers'):
                    ground_truths = [a['text'] for a in qa['answers']]
                else:
                    ground_truths = ['']
                
                # Calculate scores (take max over all ground truths)
                f1 = max([f1_score(prediction, gt) for gt in ground_truths])
                em = max([exact_match_score(prediction, gt) for gt in ground_truths])
                
                f1_scores.append(f1)
                em_scores.append(em)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
    avg_em = sum(em_scores) / len(em_scores) * 100 if em_scores else 0
    
    return avg_f1, avg_em

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='dev.json', help='Path to evaluation data')
    args = parser.parse_args()
    
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=DEVICE)
    config = checkpoint['config']
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize model
    model = TransformerQA(
        vocab_size=checkpoint['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=0.0  # No dropout during evaluation
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Get predictions
    print("Getting predictions...")
    predictions = get_predictions(model, args.data, tokenizer, config['max_len'])
    
    # Calculate scores
    print("Calculating scores...")
    f1, em = evaluate_predictions(predictions, args.data)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"F1 Score: {f1:.2f}%")
    print(f"Exact Match: {em:.2f}%")
    print("="*40)
    
    # Save predictions
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to predictions.json")

if __name__ == "__main__":
    main()