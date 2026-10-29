"""
Predict the single EXP2 instance that is missing from the SVC and RoBERTa prediction files
(issue 1769748, comment 6), using the fold whose test set contains it in train_test_split.json.

Its text is empty in the embedding csv, so the dropna() that built folds/<run>/ dropped it from
the SVC and RoBERTa test sets. The Llama run used folds_exp2/ and already predicted it; Llama is
re-predicted here for comparison.

- SVC:     no saved model, so a base SVC with the fold's exact best config (experiments_best.json)
           is refit on the same dev data (folds/<run>/data__..._gpt__oversampling_nos.joblib).
           The instance uses its stored gpt_embedding.
- RoBERTa: saved ktrain predictor in models/exp2/plm/<run>/.
- Llama:   saved LoRA adapter in models/exp2/llm-models/<run>/ on the 4-bit base model.

For SVC and RoBERTa the whole fold test set is also re-predicted and compared with the saved
EXP2 predictions, to check that the loaded/refit model reproduces the original run.

Nothing is retrained-and-saved and no existing file is modified. Each model is written as one
row (replacing an older row of the same model) to OUTPUT_PATH.

With the optional second argument "empty", RoBERTa and Llama get an empty string as input text
(the text the instance has in the embedding csv) and rows go to OUTPUT_PATH_EMPTY. SVC always uses
the stored gpt_embedding of the instance, so its prediction is the same in both modes.

Usage (from the repository root, one model per process since TF and torch both grab the GPU):
    python sources/j_predict_missing_instance_exp2.py svc|roberta|llama [empty]
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

### Config ###
ISSUE_ID = 1769748
TEXT_ID = 6

split_path = 'dataset/solution_identification_data/train_test_split.json'
best_config_path = 'experiments_best.json'
embedding_path = 'dataset/embedding_data/actual_comment_data_with_embeddings.csv'
exp2_test_path = 'dataset/solution_identification_data/folds_exp2/<run>/comment_test_data.csv'
fold_test_path = 'dataset/solution_identification_data/folds/<run>/comment_test_data.csv'
svc_data_path = 'dataset/solution_identification_data/folds/<run>/data__preprocess_none__embedding_gpt__oversampling_nos.joblib'

svc_pred_path = 'results/predictions/exp2/ml/run_<run>_with_confidence.csv'
roberta_pred_path = 'results/predictions/exp2/plm/roberta-base/folds/<run>/prediction_os_with_confidence.csv'
llama_pred_path = 'results/predictions/exp2/llm/folds/<run>/model_Llama3__bs_8__E_5__LR_1e-05__oversample_os_prediction.csv'

roberta_model_dir = 'models/exp2/plm/<run>'
llama_base_model = 'meta-llama/Meta-Llama-3-8B'
llama_adapter_path = 'models/exp2/llm-models/<run>/meta-llama/Meta-Llama-3-8B__BS_8__E_5__LR_1e-05'
LLAMA_MAX_LENGTH = 1024

OUTPUT_PATH = f'results/exp2/missing_instance/{ISSUE_ID}_{TEXT_ID}_predictions.csv'
OUTPUT_PATH_EMPTY = f'results/exp2/missing_instance/{ISSUE_ID}_{TEXT_ID}_predictions_empty_text.csv'


def find_test_fold():
    """Fold <run> == "run<run + 1>" in train_test_split.json."""
    with open(split_path, 'r') as f:
        splits = json.load(f)
    for split_run, split in splits.items():
        if any(d['issue_id'] == ISSUE_ID and d['text_id'] == TEXT_ID for d in split['test']):
            return int(split_run.replace('run', '')) - 1, split_run
    raise ValueError(f"{ISSUE_ID}/{TEXT_ID} is not in any test split of {split_path}")


def get_instance(run):
    test_df = pd.read_csv(exp2_test_path.replace('<run>', f'{run}'))
    row = test_df[(test_df['issue_id'] == ISSUE_ID) & (test_df['text_id'] == TEXT_ID)]
    if len(row) != 1:
        raise ValueError(f"Expected 1 row for {ISSUE_ID}/{TEXT_ID} in fold {run}, found {len(row)}")
    return row.iloc[0]


def get_saved_prediction(path, pred_col, prob_col):
    """Saved EXP2 prediction of the instance, None when it is missing from the file."""
    df = pd.read_csv(path)
    row = df[(df['issue_id'] == ISSUE_ID) & (df['text_id'] == TEXT_ID)]
    if row.empty:
        return None, None
    return int(row.iloc[0][pred_col]), float(row.iloc[0][prob_col])


def reproduction_check(saved_path, pred_col, new_preds):
    saved = pd.read_csv(saved_path)[pred_col].astype(int).to_numpy()
    if len(saved) != len(new_preds):
        return f"length mismatch: saved {len(saved)} vs new {len(new_preds)}"
    return f"{int((saved == np.asarray(new_preds).astype(int)).sum())}/{len(saved)} match saved predictions"


def predict_svc(run, instance):
    import joblib
    from sklearn.svm import SVC

    best = json.load(open(best_config_path))['best_model_config'][str(run)]['SVC']
    print(f"[SVC] Fold {run} best config: {best}", flush=True)
    if best['preprocess'] != 'none' or best['embedding'] != 'gpt' or best['oversampling']:
        raise ValueError(f"Unexpected SVC setup for fold {run}: {best}")

    data_dict = joblib.load(svc_data_path.replace('<run>', f'{run}'))

    emb_df = pd.read_csv(embedding_path, usecols=['issue_id', 'text_id', 'gpt_embedding'])
    emb_row = emb_df[(emb_df['issue_id'] == ISSUE_ID) & (emb_df['text_id'] == TEXT_ID)]
    X_instance = np.array([ast.literal_eval(emb_row.iloc[0]['gpt_embedding'])])

    # Same as getBaseModel("SVC") and set_ml_seed(42) in d_mlm_experiments.py
    np.random.seed(42)
    clf = SVC(random_state=42, probability=True).set_params(**best['config'])
    print(f"[SVC] Fitting on dev data {data_dict['X_dev_embedding'].shape}...", flush=True)
    clf.fit(data_dict['X_dev_embedding'], data_dict['y_dev'])

    positive_idx = list(clf.classes_).index(1)
    check = reproduction_check(
        svc_pred_path.replace('<run>', f'{run}'),
        f'none_gpt_nos_SVC_{run}',
        clf.predict(data_dict['X_test_embedding'])
    )

    pred = int(clf.predict(X_instance)[0])
    prob = float(clf.predict_proba(X_instance)[0, positive_idx])
    saved_pred, saved_prob = get_saved_prediction(
        svc_pred_path.replace('<run>', f'{run}'), f'none_gpt_nos_SVC_{run}', f'none_gpt_nos_SVC_{run}_confidence'
    )
    return {
        'model': 'SVC',
        'model_source': f"refit base SVC on {svc_data_path.replace('<run>', f'{run}')}",
        'config': json.dumps(best),
        'prediction': pred,
        'positive_probability': prob,
        'saved_exp2_prediction': saved_pred,
        'saved_exp2_positive_probability': saved_prob,
        'fold_reproduction_check': check,
    }


def predict_roberta(run, instance):
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import ktrain

    model_dirs = [d for d in os.listdir(roberta_model_dir.replace('<run>', f'{run}')) if d.startswith('roberta-base__')]
    if len(model_dirs) != 1:
        raise ValueError(f"Expected 1 roberta-base model for fold {run}, found {model_dirs}")
    model_path = os.path.join(roberta_model_dir.replace('<run>', f'{run}'), model_dirs[0])

    print(f"[RoBERTa] Loading predictor from {model_path}", flush=True)
    predictor = ktrain.load_predictor(model_path)
    positive_idx = [str(c) for c in predictor.get_classes()].index('1')

    fold_texts = pd.read_csv(fold_test_path.replace('<run>', f'{run}'))['text'].tolist()
    print(f"[RoBERTa] Re-predicting the {len(fold_texts)} fold test texts for the reproduction check...", flush=True)
    check = reproduction_check(
        roberta_pred_path.replace('<run>', f'{run}'),
        'roberta-base_os_prediction',
        predictor.predict(fold_texts)
    )

    pred = int(predictor.predict([instance['text']])[0])
    prob = float(np.asarray(predictor.predict_proba([instance['text']]))[0, positive_idx])
    saved_pred, saved_prob = get_saved_prediction(
        roberta_pred_path.replace('<run>', f'{run}'), 'roberta-base_os_prediction', 'roberta-base_os_confidence'
    )
    return {
        'model': 'roberta-base',
        'model_source': model_path,
        'config': json.dumps(json.load(open(best_config_path))['best_model_config'][str(run)]['roberta-base']),
        'prediction': pred,
        'positive_probability': prob,
        'saved_exp2_prediction': saved_pred,
        'saved_exp2_positive_probability': saved_prob,
        'fold_reproduction_check': check,
    }


def predict_llama(run, instance):
    import torch
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
    from utils import set_llm_seed

    set_llm_seed(42)
    adapter_path = llama_adapter_path.replace('<run>', f'{run}')

    # Load the base model exactly as during training, then attach the LoRA adapter (not merged)
    print(f"[Llama] Loading base model {llama_base_model} (4-bit)...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        llama_base_model,
        quantization_config=quantization_config,
        num_labels=2,
        device_map="cuda:0",
    )
    base_model = prepare_model_for_kbit_training(base_model)
    print(f"[Llama] Attaching LoRA adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(llama_base_model, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.eval()

    inputs = tokenizer([instance['text']], return_tensors="pt", padding=True, truncation=True, max_length=LLAMA_MAX_LENGTH)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        probabilities = F.softmax(model(**inputs)['logits'].float(), dim=1)

    saved_pred, saved_prob = get_saved_prediction(
        llama_pred_path.replace('<run>', f'{run}'),
        'meta-llama/Meta-Llama-3-8Bbs_8__E_5__LR_1e-05__oversample_os_prediction',
        'positive_probability'
    )
    return {
        'model': 'Llama3-8B',
        'model_source': adapter_path,
        'config': json.dumps(json.load(open(best_config_path))['llm_model_conf']),
        'prediction': int(torch.argmax(probabilities, dim=1)[0]),
        'positive_probability': float(probabilities[0, 1]),
        'saved_exp2_prediction': saved_pred,
        'saved_exp2_positive_probability': saved_prob,
        'fold_reproduction_check': 'not run (instance already in the saved fold predictions)',
    }


def save_row(row, output_path):
    """One row per model; a re-run of a model replaces its previous row."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_df = pd.DataFrame([row])
    if os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        new_df = pd.concat([old_df[old_df['model'] != row['model']], new_df], ignore_index=True)
    new_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    predictors = {'svc': predict_svc, 'roberta': predict_roberta, 'llama': predict_llama}
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in predictors or sys.argv[2:] not in ([], ['empty']):
        sys.exit(f"Usage: python {sys.argv[0]} {{{'|'.join(predictors)}}} [empty]")
    empty_text = sys.argv[2:] == ['empty']

    run, split_run = find_test_fold()
    instance = get_instance(run).copy()
    if empty_text:
        instance['text'] = ''
    print(f"{ISSUE_ID}/{TEXT_ID} is in the test set of {split_run} -> fold {run} | label {instance['label']}", flush=True)

    result = predictors[sys.argv[1]](run, instance)
    row = {
        'issue_id': ISSUE_ID,
        'text_id': TEXT_ID,
        'text': instance['text'],
        'input_mode': 'empty_text' if empty_text else 'comment_text',
        'label': int(instance['label']),
        'split_json_run': split_run,
        'fold': run,
        **result,
    }
    print("=" * 50)
    for k, v in row.items():
        print(f"{k}: {v}")
    print("=" * 50)
    save_row(row, OUTPUT_PATH_EMPTY if empty_text else OUTPUT_PATH)
