import numpy as np
import pandas as pd
import torch
import os

import json
import socket
import threading
import pynvml
import time

from utils import append_row_to_csv, calculate_results, get_experiment_config, prepare_data, print_current_time, process_run, set_llm_seed

from utils import dev_data_path, test_data_path, train_data_path, val_data_path, train_os_data_path, dev_os_df_path, mapped_comment_data, train_test_comment_data, comment_data_llama_prediction_exp2

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Sub-folder for this rerun so the earlier outputs, models, results and predictions are kept
RERUN_TAG = os.environ.get("RERUN_TAG", "rerun_eval_loss")

# EXP2-only folds built from labeled_comment_data.csv + train_test_split.json, kept apart from the shared
# folds/ (rebuilt with embeddings, where dropna() loses a comment and creation_time/comment_id are missing)
exp2_folds_dir = 'dataset/solution_identification_data/folds_exp2/<run>'
exp2_fold_paths = [
    f'{exp2_folds_dir}/{os.path.basename(p)}'
    for p in (dev_data_path, test_data_path, train_data_path, val_data_path, train_os_data_path, dev_os_df_path)
]
exp2_dev_data_path, exp2_test_data_path, exp2_train_data_path, exp2_val_data_path, exp2_train_os_data_path, exp2_dev_os_df_path = exp2_fold_paths


cost_path = f"results/exp2/{RERUN_TAG}/llm_fine_tuning/computational_cost.csv"
log_history_dir = f"results/exp2/{RERUN_TAG}/llm_fine_tuning/log_history"


class GpuMonitor:
    """Samples power draw and utilization of the GPU used by this process with NVML in the background,
    and reads the GPU's cumulative energy counter for the energy used"""

    def __init__(self, interval_s=2):
        self.interval_s = interval_s
        self.samples = []
        self.handle = None
        self.start_energy_mj = None
        self.end_energy_mj = None

    def _find_handle(self):
        # CUDA_VISIBLE_DEVICES may not follow NVML's device order, so match the GPU running this process first
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            if any(p.pid == os.getpid() for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)):
                return handle
        return pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))

    def _read_energy_mj(self):
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except pynvml.NVMLError:
            return None

    def __enter__(self):
        self.start = time.perf_counter()
        try:
            pynvml.nvmlInit()
            self.handle = self._find_handle()
        except (pynvml.NVMLError, ValueError) as error:
            print(f"NVML unavailable ({error}), GPU power/utilization will not be recorded")
            return self
        self.start_energy_mj = self._read_energy_mj()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                self.samples.append((power_w, utilization))
            except pynvml.NVMLError:
                pass
            self.stop_event.wait(self.interval_s)

    def __exit__(self, *exc):
        self.duration_s = time.perf_counter() - self.start
        if self.handle is not None:
            self.stop_event.set()
            self.thread.join(timeout=10)
            self.end_energy_mj = self._read_energy_mj()
        return False

    def summary(self, prefix):
        powers = [p for p, _ in self.samples]
        utilizations = [u for _, u in self.samples]
        avg_power_w = float(np.mean(powers)) if powers else float('nan')
        if self.start_energy_mj is not None and self.end_energy_mj is not None:
            energy_kwh = (self.end_energy_mj - self.start_energy_mj) / 3.6e9
        else:
            energy_kwh = avg_power_w * self.duration_s / 3.6e6
        return {
            f'{prefix}_avg_gpu_power_w': round(avg_power_w, 1),
            f'{prefix}_max_gpu_power_w': round(max(powers), 1) if powers else float('nan'),
            f'{prefix}_gpu_energy_kwh': round(energy_kwh, 4),
            f'{prefix}_avg_gpu_utilization_pct': round(float(np.mean(utilizations)), 1) if utilizations else float('nan'),
            f'{prefix}_n_gpu_samples': len(powers),
        }


def get_dir_size_mb(path, prefix=''):
    if path is None or not os.path.isdir(path):
        return float('nan')
    return round(sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.startswith(prefix)) / 1024 ** 2, 2)


def get_peak_gpu_memory_gb():
    if not torch.cuda.is_available():
        return 0, 0
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2), round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)


def prepare_exp2_folds(run):
    if all(os.path.exists(p.replace('<run>', f'{run}')) for p in exp2_fold_paths):
        return
    data = pd.read_csv(mapped_comment_data)
    with open(train_test_comment_data, 'r') as f:
        indexes = json.load(f)
    process_run(run, data, indexes, exp2_fold_paths)

    # The folds must match train_test_split.json (fold <run> == "run<run + 1>")
    for split, path in (('train', exp2_dev_data_path), ('test', exp2_test_data_path)):
        expected = [(d['issue_id'], d['text_id']) for d in indexes[f'run{run + 1}'][split]]
        actual = list(pd.read_csv(path.replace('<run>', f'{run}'))[['issue_id', 'text_id']].itertuples(index=False, name=None))
        if expected != actual:
            raise ValueError(f"Fold {run} {split} split does not match {train_test_comment_data}")

torch.cuda.empty_cache()
import torch.nn.functional as F

from datasets import DatasetDict, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer

def separator_print(text, chr='-', num=50):
    print(chr * num)
    print(f"\n{text}\n")
    print(chr * num)


def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    ba = balanced_accuracy_score(labels, predictions)

    return {
        'f1_score': f1,
        'accuracy': accuracy,
        'balanced_accuracy': ba,
        'precision': precision,
        'recall': recall,
    }



def train_evaluate_llm(run, model_name, dev_df, train_df, val_df, test_df, bs, lr, e, save_loc=None):
    set_llm_seed(42)

    ### Innitial Params ###
    max_length = 1024
    token_batch_size = 8
    logging_steps = 5
    weight_decay = 0.01

    target_column = 'label'

    zs_column = f'ZS_predictions_{bs}_{e}'
    ft_column = f'FT_predictions_{bs}_{e}'

    ### Convert to dataset ###
    dataset_train = Dataset.from_pandas(train_df[['text', target_column]])
    dataset_val = Dataset.from_pandas(val_df[['text', target_column]])
    dataset_test = Dataset.from_pandas(test_df[['text', target_column]])
    dataset_dev = Dataset.from_pandas(dev_df[['text', target_column]])

    # Combine them into a single DatasetDict
    dataset = DatasetDict({
        'dev': dataset_dev,
        'test': dataset_test,
    })

    print("\nDataset Format: ", dataset)

    ### Define Class Weight ###
    class_weights = (1 / train_df[target_column].value_counts(normalize=True).sort_index()).tolist()
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()
    print(f"\n\nClass Weight: {class_weights}")

    ### Model Load ###
    model_load_start = time.perf_counter()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        num_labels=2,
        device_map='auto',
        # cache_dir="/scratch/msun12/models"
    )

    ### Add LoRA Adaptation ###
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='SEQ_CLS'
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model_load_s = time.perf_counter() - model_load_start

    ### Get Tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    ### Preprocess Data ###
    def data_preprocesing(row):
        return tokenizer(row['text'], padding=True, truncation=True, max_length=max_length)

    tokenized_data = dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
    tokenized_data.set_format("torch")

    separator_print("Tokenized Data")
    print(tokenized_data)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    # wandb.init(
    #     project="solution_loc_llm",
    # )   

    training_args = TrainingArguments(
        output_dir=f'output/exp2/{RERUN_TAG}/{run}',
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=e,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        # report_to="wandb",
        # metric_for_best_model='f1_score',
        # greater_is_better=True,
        report_to='none',
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['dev'],
        eval_dataset=tokenized_data['test'], 
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    separator_print("TRAIN DATASET")
    print(tokenized_data['dev'])
    print(tokenized_data['test'])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()

    with GpuMonitor() as train_gpu_monitor:
        train_result = trainer.train()

    train_wall_s = time.perf_counter() - train_start
    train_peak_allocated_gb, train_peak_reserved_gb = get_peak_gpu_memory_gb()

    separator_print("TRAIN RESULT")
    print(train_result)

    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        separator_print("Saving Model ...")
        
        model.save_pretrained(save_loc)
        tokenizer.save_pretrained(save_loc)

    ### Evaluation ###
    def generate_predictions(model, df_test, bs, max_length):
        sentences = df_test['text'].tolist()
        batch_size = bs

        all_predictions = []
        all_confidences = []
        all_probabilities = []

        separator_print(f"Generating Predictions for {len(sentences)} sentences with batch size {batch_size}")

        for i in range(0, len(sentences), batch_size):
            print(f"Processing batch {i // batch_size + 1} of {len(sentences) // batch_size + 1}")
            batch_sentences = sentences[i:i + batch_size]

            inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )

            inputs = {
                k: v.to('cuda' if torch.cuda.is_available() else 'cpu')
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**inputs)

                # Convert logits to class probabilities
                probabilities = F.softmax(outputs['logits'].float(), dim=1)

                # Predicted class and its confidence
                confidences, predictions = torch.max(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

        return (
            np.array(all_predictions),
            np.array(all_confidences),
            np.array(all_probabilities)
        )

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inference_start = time.perf_counter()

    with GpuMonitor() as inference_gpu_monitor:
        y_pred, y_confidence, y_probability = generate_predictions(model, test_df, bs, max_length)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    inference_wall_s = time.perf_counter() - inference_start
    inference_peak_allocated_gb, inference_peak_reserved_gb = get_peak_gpu_memory_gb()

    ### Computational Cost ###
    # train_runtime includes the per-epoch evaluation on the test set, so the pure training time is reported separately
    eval_runtimes = [log['eval_runtime'] for log in trainer.state.log_history if 'eval_runtime' in log]
    train_runtime_s = train_result.metrics.get('train_runtime', train_wall_s)
    n_trainable_params, n_total_params = model.get_nb_trainable_parameters()

    # Token usage: content tokens (no padding) vs. tokens fed to the model (with padding)
    def count_untruncated_tokens(texts):
        return [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]

    train_token_lengths = count_untruncated_tokens(dev_df['text'].tolist())
    test_token_lengths = count_untruncated_tokens(test_df['text'].tolist())

    n_train_tokens = sum(min(n, max_length) for n in train_token_lengths)
    n_train_tokens_padded = sum(len(ids) for ids in tokenized_data['dev']['input_ids'])
    n_test_tokens = sum(min(n, max_length) for n in test_token_lengths)
    n_eval_tokens_padded = sum(len(ids) for ids in tokenized_data['test']['input_ids'])
    # generate_predictions pads each batch to its longest comment
    n_inference_tokens_padded = sum(
        max(min(n, max_length) for n in test_token_lengths[i:i + bs]) * len(test_token_lengths[i:i + bs])
        for i in range(0, len(test_token_lengths), bs)
    )

    cost_row = {
        'run': run,
        'model_name': model_name,
        'batch_size': bs,
        'epoch': e,
        'learning_rate': lr,
        'hostname': socket.gethostname(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'n_train_comments': len(dev_df),
        'n_train_steps': train_result.global_step,
        'n_trainable_params': n_trainable_params,
        'n_total_params': n_total_params,
        'train_runtime_s': round(train_runtime_s, 2),
        'eval_runtime_total_s': round(sum(eval_runtimes), 2),
        'train_only_runtime_s': round(train_runtime_s - sum(eval_runtimes), 2),
        'train_wall_s': round(train_wall_s, 2),
        'train_peak_gpu_allocated_gb': train_peak_allocated_gb,
        'train_peak_gpu_reserved_gb': train_peak_reserved_gb,
        **train_gpu_monitor.summary('train'),
        'model_load_s': round(model_load_s, 2),
        'adapter_size_mb': get_dir_size_mb(save_loc, prefix='adapter_model'),
        'saved_model_dir_size_mb': get_dir_size_mb(save_loc),
        'max_length': max_length,
        'n_truncated_train_comments': sum(n > max_length for n in train_token_lengths),
        'n_train_tokens_per_epoch': n_train_tokens,
        'n_train_tokens_padded_per_epoch': n_train_tokens_padded,
        'n_train_tokens_total': n_train_tokens * e,
        'n_train_tokens_padded_total': n_train_tokens_padded * e,
        'n_epoch_evals': len(eval_runtimes),
        'n_eval_tokens_total': n_test_tokens * len(eval_runtimes),
        'n_eval_tokens_padded_total': n_eval_tokens_padded * len(eval_runtimes),
        'n_test_comments': len(test_df),
        'n_truncated_test_comments': sum(n > max_length for n in test_token_lengths),
        'n_test_tokens': n_test_tokens,
        'n_inference_tokens_padded': n_inference_tokens_padded,
        'inference_tokens_per_s': round(n_test_tokens / inference_wall_s, 2),
        'inference_wall_s': round(inference_wall_s, 2),
        'inference_s_per_comment': round(inference_wall_s / len(test_df), 4),
        'inference_peak_gpu_allocated_gb': inference_peak_allocated_gb,
        'inference_peak_gpu_reserved_gb': inference_peak_reserved_gb,
        **inference_gpu_monitor.summary('inference'),
    }

    separator_print("COMPUTATIONAL COST")
    print(cost_row)

    if not os.path.isfile(cost_path):
        append_row_to_csv(cost_path, list(cost_row.keys()))
    append_row_to_csv(cost_path, list(cost_row.values()))
    print(f"\nComputational cost saved to {cost_path}\n")

    # Per-epoch loss, eval metrics and runtimes
    os.makedirs(log_history_dir, exist_ok=True)
    log_history_path = os.path.join(log_history_dir, f"fold_{run}__BS_{bs}__E_{e}__LR_{lr}.json")
    with open(log_history_path, 'w') as f:
        json.dump({'best_model_checkpoint': trainer.state.best_model_checkpoint, 'log_history': trainer.state.log_history}, f, indent=2)
    print(f"Training log history saved to {log_history_path}\n")

    return y_pred, y_confidence, y_probability


def llama_experiments(run, mod, result_path, prediction_path, oversample=True, hyperparam_tuning=False):
    exp_config = get_experiment_config()
    
    ### Data Retrieve ###
    dev_df = pd.read_csv(exp2_dev_data_path.replace('<run>', f'{run}'))
    test_df = pd.read_csv(exp2_test_data_path.replace('<run>', f'{run}'))
    train_df = pd.read_csv(exp2_train_data_path.replace('<run>', f'{run}'))
    val_df = pd.read_csv(exp2_val_data_path.replace('<run>', f'{run}'))
    train_df_os = pd.read_csv(exp2_train_os_data_path.replace('<run>', f'{run}'))
    dev_df_os = pd.read_csv(exp2_dev_os_df_path.replace('<run>', f'{run}'))

    model_name = mod['model_name']
    print(f"\n\n\nEvaluating Model: {model_name}\n\n\n")

    if hyperparam_tuning:
        if oversample:
            train_df = train_df_os
            dev_df = dev_df_os
    else:
        oversample = exp_config['llm_model_conf']['params']['oversample']
        if oversample:
            train_df = train_df_os
            dev_df = dev_df_os

    dev_df = dev_df[["issue_id","text_id","text","label","creation_time","comment_id"]]
    test_df = test_df[["issue_id","text_id","text","label","creation_time","comment_id"]]
    train_df = train_df[["issue_id","text_id","text","label","creation_time","comment_id"]]
    val_df = val_df[["issue_id","text_id","text","label","creation_time","comment_id"]]

    ### Print Data Stats ###
    print(f"Dev Data Length: {dev_df}")
    print(f"Dev data distribution: {dev_df[target_column].value_counts(normalize=True)}")
    print(f"Dev data distribution: {dev_df[target_column].value_counts()}")

    print(f"Test Data Length: {test_df}")
    print(f"Test data distribution: {test_df[target_column].value_counts(normalize=True)}")
    print(f"Test data distribution: {test_df[target_column].value_counts()}")

    print(f"Train Data Length: {train_df}")
    print(f"Train data distribution: {train_df[target_column].value_counts(normalize=True)}")
    print(f"Train data distribution: {train_df[target_column].value_counts()}")

    print(f"Val Data Length: {val_df}")
    print(f"Val data distribution: {val_df[target_column].value_counts(normalize=True)}")
    print(f"Val data distribution: {val_df[target_column].value_counts()}")

    if hyperparam_tuning:
        BS, E, LR = hyperparameter_tuning(mod, model_name, oversample, run, train_df, val_df)
    else:
        BS = exp_config['llm_model_conf']['params']['batch_size']
        E = exp_config['llm_model_conf']['params']['epoch']
        LR = exp_config['llm_model_conf']['params']['learning_rate']

    separator_print(f"Run: {run}, Oversample: {oversample}, Batch Size: {BS}, Epoch: {E}, Learning Rate: {LR}")

    model_save_path = f"models/exp2/{RERUN_TAG}/llm/{run}/{model_name.replace(' ', '_')}__BS_{BS}__E_{E}__LR_{LR}"
    y_pred, y_confidence, y_probability = train_evaluate_llm(run, model_name, dev_df, train_df, val_df, test_df, BS, LR, E, save_loc=model_save_path)
    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, test_df[target_column])

    result_ft = [
        run, oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN
    ]

    print("=" * 50, "Fine tuned", "=" * 50)
    print(f"Batch: {BS} || E: {E} || LR: {LR} || Oversample: {oversample}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print("=" * 50)

    append_row_to_csv(result_path, result_ft)
    print(f"\nResults succesfully saved to {result_path}\n\n")

    os.makedirs(prediction_path, exist_ok=True)

    test_df_copy = test_df.copy()

    file_name = model_name + f"bs_{BS}__E_{E}__LR_{LR}__oversample_{'os' if oversample else 'nos'}_prediction"
    
    test_df_copy[file_name] = y_pred
    test_df_copy['confidence'] = y_confidence
    test_df_copy['positive_probability'] = y_probability

    path = os.path.join(prediction_path, f"model_Llama3__bs_{BS}__E_{E}__LR_{LR}__oversample_{'os' if oversample else 'nos'}_prediction.csv")
    test_df_copy.to_csv(path, index=False)
    print(f"\n\nPredictions succesfully saved to {path}\n\n")


def hyperparameter_tuning(mod, model_name, oversample, run, train_df, val_df):
    BS = -1
    LR = -1
    E = -1
    F1 = -1
    for bs in mod['params']['batch_size']:
        for lr in mod['params']['learning_rate']:
            for e in mod['params']['epoch']:
                y_pred, _, _ = train_evaluate_llm(run, model_name, train_df, val_df, bs, lr, e)
                FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, val_df[target_column])

                print("=" * 50, "Inside Validation", "=" * 50)
                print(f"batch_sizes: {bs}")
                print(f"total_epochs: {e}")
                print(f"learning_rate: {lr}")
                print(f"Model name: {model_name}")
                print(f"Oversampling: {oversample}")
                print(f"Run: {run}")
                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"True Positives (TP): {TP}")
                print(f"False Positives (FP): {FP}")
                print(f"True Negatives (TN): {TN}")
                print(f"False Negatives (FN): {FN}")
                print("=" * 50)

                if f1 > F1:
                    BS = bs
                    LR = lr
                    E = e
                    F1 = f1
    print("-" * 50)
    print(f"Completed Validating with Max F1 Score: {F1}")
    print(f"Batch Size:{BS}")
    print(f"Epoch:{E}")
    print(f"Learning Rate:{LR}")
    print("-" * 50)
    return BS, E, LR


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights,
                                              dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    print_current_time()

    # runs = [0,1,2,3,4,5,6,7,8,9]
    runs = [int(os.environ.get("RUN_ID", "7"))]

    exp_config = get_experiment_config()
    # prepare_data(runs, mapped_comment_data, train_test_comment_data)

    target_column = 'label'

    res_path = f"results/exp2/{RERUN_TAG}/llm_fine_tuning/comment_data_results.csv"
    result = ["run", "oversample", "batch_size", "epoch", "initial_learning_Rate", "model_name", "accuracy",
                "precision", "recall", "f1", "TP", "FP", "TN", "FN"]

    append_row_to_csv(res_path, result)

    for run in runs:
        prepare_exp2_folds(run)
        for model in exp_config['llm_model']:
            pred_path = f"results/predictions/exp2/{RERUN_TAG}/llm/folds/{run}/"
            llama_experiments(run, model, res_path, pred_path, hyperparam_tuning=False)

    print_current_time()
