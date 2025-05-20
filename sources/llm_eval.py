import argparse
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine‐tuned HuggingFace model on new data"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='models/llm/0/meta-llama/Meta-Llama-3-8B__BS_8__E_5__LR_1e-05',
        help="Path to directory where model & tokenizer were saved",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='results/predictions/llm/folds/0/llm_eval_predictions.csv',
        help="Path to CSV file containing new data (must have 'text' & 'label')",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max token length (should match what you trained with)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='results/predictions/llm/folds/0/llm_eval_predictions.csv',
        help=(
            "Where to save predictions CSV. "
            "If not set, saves alongside input as `<basename>_predictions.csv`"
        ),
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns.")
    return df


def make_dataloader(texts, labels, tokenizer, max_length, batch_size, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"], enc["attention_mask"], torch.tensor(labels)
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in tqdm(dataloader, desc="Infer"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_and_print_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Precision:          {prec:.4f}")
    print(f"Recall:             {rec:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print("==========================\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load model + tokenizer
    print(f"Loading tokenizer & model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    tokenizer.pad_token_id   = tokenizer.eos_token_id
    tokenizer.pad_token      = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(device)

    # 2) load data
    df = load_data(args.data_path)
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 3) make DataLoader
    loader = make_dataloader(
        texts,
        labels,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    # 4) inference
    y_true, y_pred = evaluate(model, loader, device)

    # 5) metrics
    compute_and_print_metrics(y_true, y_pred)

    # 6) save predictions
    df["predicted_label"] = y_pred
    out_path = (
        args.output_path
        if args.output_path
        else os.path.splitext(args.data_path)[0] + "_predictions.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
