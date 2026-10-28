"""
Re-run inference with the saved EXP2 Llama-3 fine-tuned models on the instances whose
prediction differs from the LLAMA-FT column of results_analysis/ensembled_models.xlsx.

For every fold in RUNS the model is loaded once and all mismatched instances of that
fold are inferred. Each instance gets at most MAX_ATTEMPTS tries; every attempt runs
inference only on the still unmatched instances (in fold order, bs=ORIGINAL_BS).
When the new prediction equals the xlsx prediction, the fold prediction file and the
fold row of the results csv are updated. Every response is saved to RESPONSES_PATH.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

from utils import calculate_results, print_current_time, set_llm_seed, comment_data_llama_prediction_exp2

### Config ###
RUNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # only these models are available in models/exp2/llm for now
MAX_ATTEMPTS = 3  # binary labels: 3 identical answers means no match, otherwise it already matched
MAX_LENGTH = 1024
ORIGINAL_BS = 8

model_name = "meta-llama/Meta-Llama-3-8B"
model_dir = "models/exp2/llm/<run>/meta-llama/Meta-Llama-3-8B__BS_8__E_5__LR_1e-05"
prediction_file = "model_Llama3__bs_8__E_5__LR_1e-05__oversample_os_prediction.csv"
result_path = "results/exp2/llm_fine_tuning/comment_data_results.csv"
mismatch_path = "results_analysis/llm_exp2_vs_ensembled_prediction_diffs.csv"
RESPONSES_PATH = "results/exp2/llm_fine_tuning/reinference_159_mismatches.csv"

XLSX_COL = 'ensembled_xlsx_LLAMA-FT_pred'


def separator_print(text, chr='-', num=50):
    print(chr * num)
    print(f"\n{text}\n")
    print(chr * num)


def load_model(run):
    """Load the base model exactly as it was during training, then attach the LoRA adapter."""
    adapter_path = model_dir.replace('<run>', f'{run}')
    print(f"[Run {run}] Loading base model {model_name} (4-bit)...", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        num_labels=2,
        device_map="cuda:0",
    )
    print(f"[Run {run}] Base model loaded, preparing for k-bit...", flush=True)
    base_model = prepare_model_for_kbit_training(base_model)

    print(f"[Run {run}] Attaching LoRA adapter from {adapter_path}...", flush=True)

    # Do not merge: merging LoRA weights into a 4-bit model changes the predictions
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"[Run {run}] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model.eval()
    print(f"[Run {run}] Model and tokenizer ready", flush=True)
    return model, tokenizer


def generate_predictions(model, tokenizer, sentences, batch_size):
    """Same inference loop as the training script."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_predictions, all_confidences, all_probabilities = [], [], []
    n_batches = (len(sentences) + batch_size - 1) // batch_size
    print(f"Inference on {len(sentences)} texts in {n_batches} batches (bs={batch_size})", flush=True)

    for i in range(0, len(sentences), batch_size):
        batch_num = i // batch_size + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == n_batches:
            print(f"  batch {batch_num}/{n_batches}", flush=True)
        inputs = tokenizer(
            sentences[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs['logits'].float(), dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    print("Inference done", flush=True)
    return np.array(all_predictions), np.array(all_confidences), np.array(all_probabilities)


def update_result_row(run, fold_df, pred_col):
    """Recompute the metrics of one fold and replace its row in the results csv."""
    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(fold_df[pred_col], fold_df['label'])

    print(f"Updating fold row of run {run} in {result_path}...", flush=True)
    results = pd.read_csv(result_path)
    idx = results.index[results['run'] == run]
    results.loc[idx, ['accuracy', 'precision', 'recall', 'f1', 'TP', 'FP', 'TN', 'FN']] = \
        [accuracy, precision, recall, f1, TP, FP, TN, FN]
    results.to_csv(result_path, index=False)

    print(f"Updated results of run {run}: Acc {accuracy} | P {precision} | R {recall} | F1 {f1} "
          f"| TP {TP} FP {FP} TN {TN} FN {FN}")


def reinfer_fold(run, fold_mismatch, responses):
    fold_pred_path = os.path.join(comment_data_llama_prediction_exp2.replace('<run>', f'{run}'), prediction_file)
    print(f"[Run {run}] Reading fold predictions from {fold_pred_path}", flush=True)
    fold_df = pd.read_csv(fold_pred_path)
    pred_col = [c for c in fold_df.columns if '_prediction' in c][0]

    # position of every mismatched instance in the fold test set
    key_to_pos = {(r.issue_id, r.text_id): i for i, r in enumerate(fold_df.itertuples())}
    targets = {(r['issue_id'], r['text_id']): int(r[XLSX_COL]) for r in fold_mismatch.to_dict('records')}
    pending = sorted(targets, key=lambda k: key_to_pos[k])  # keep the original fold order
    print(f"[Run {run}] Fold test set size {len(fold_df)} | {len(pending)} instances to re-infer", flush=True)

    model, tokenizer = load_model(run)
    texts = fold_df['text'].tolist()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            break

        print(f"[Run {run}] Attempt {attempt}/{MAX_ATTEMPTS} started ({len(pending)} pending)", flush=True)
        print_current_time()
        # only the still unmatched instances, in original fold order
        pending_texts = [texts[key_to_pos[key]] for key in pending]
        preds, confs, probs = generate_predictions(model, tokenizer, pending_texts, ORIGINAL_BS)
        out = {key: (preds[i], confs[i], probs[i]) for i, key in enumerate(pending)}

        still_pending = []
        for key in pending:
            pred, conf, prob = out[key]
            resp = responses[key]
            resp[f'attempt_{attempt}_pred'] = int(pred)
            resp[f'attempt_{attempt}_positive_probability'] = float(prob)
            resp['attempts_run'] = attempt

            if int(pred) == targets[key]:
                resp['matched'] = True
                resp['matched_attempt'] = attempt
                resp['new_pred'] = int(pred)
                pos = key_to_pos[key]
                fold_df.loc[pos, pred_col] = int(pred)
                fold_df.loc[pos, 'confidence'] = float(conf)
                fold_df.loc[pos, 'positive_probability'] = float(prob)
            else:
                still_pending.append(key)

        print(f"Run {run} | attempt {attempt} | matched {len(pending) - len(still_pending)} "
              f"| still unmatched {len(still_pending)}")
        pending = still_pending

    n_matched = len(targets) - len(pending)
    separator_print(f"Run {run}: {n_matched}/{len(targets)} instances matched the xlsx prediction")

    if n_matched:
        fold_df.to_csv(fold_pred_path, index=False)
        print(f"Predictions updated in {fold_pred_path}")
        update_result_row(run, fold_df, pred_col)

    else:
        print(f"[Run {run}] No matches, prediction file and results left unchanged", flush=True)

    print(f"[Run {run}] Releasing model and GPU memory", flush=True)
    del model
    torch.cuda.empty_cache()


def save_responses(responses):
    df = pd.DataFrame(list(responses.values()))
    os.makedirs(os.path.dirname(RESPONSES_PATH), exist_ok=True)
    df.to_csv(RESPONSES_PATH, index=False)
    print(f"Responses saved to {RESPONSES_PATH}")


if __name__ == "__main__":
    print_current_time()
    set_llm_seed(42)

    print(f"Reading mismatches from {mismatch_path}", flush=True)
    mismatch_df = pd.read_csv(mismatch_path)
    print(f"Loaded {len(mismatch_df)} mismatched instances; per fold: "
          f"{mismatch_df['prediction_fold_dir'].value_counts().sort_index().to_dict()}", flush=True)

    responses = {}
    for r in mismatch_df.to_dict('records'):
        responses[(r['issue_id'], r['text_id'])] = {
            'test_fold_in_split_json': r['test_fold_in_split_json'],
            'prediction_fold_dir': r['prediction_fold_dir'],
            'issue_id': r['issue_id'],
            'text_id': r['text_id'],
            'label': r['label'],
            'exp2_llm_pred': r['exp2_llm_pred'],
            XLSX_COL: r[XLSX_COL],
            'status': 'inferred' if r['prediction_fold_dir'] in RUNS else 'skipped_no_model',
            'attempts_run': 0,
            'matched': False,
            'matched_attempt': None,
            'new_pred': None,
        }

    for run in RUNS:
        fold_mismatch = mismatch_df[mismatch_df['prediction_fold_dir'] == run]
        separator_print(f"Run {run}: {len(fold_mismatch)} mismatched instances")
        if len(fold_mismatch):
            reinfer_fold(run, fold_mismatch, responses)
        save_responses(responses)  # save after each fold so progress is not lost
        print_current_time()

    n_total_matched = sum(r['matched'] for r in responses.values())
    separator_print(f"Done: {n_total_matched}/{len(responses)} instances matched the xlsx prediction", chr='=')
    print_current_time()
