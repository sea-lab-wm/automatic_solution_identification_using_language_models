import os
import wandb
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F

from datasets import DatasetDict, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score
from transformers import Trainer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import Dataset, DatasetDict

from utils import append_row_to_csv, calculate_results, set_seed, set_llm_seed

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain
from ktrain import text
import os
import pandas as pd

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


def get_predictions(X_test, model, tokenizer, max_length=1024, batch_size=8):
    model.eval()

    all_outputs = []
    device = next(model.parameters()).device

    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_outputs.extend(predictions.cpu().numpy())

    return all_outputs


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


def fine_tune_model(train_df, model, tokenizer, oversample, BS, LR, E, model_to_save_path):
    set_llm_seed(42)

    max_length = 1024
    token_batch_size = 8
    logging_steps = 5
    weight_decay = 0.01

    ### Convert to dataset ###
    dataset_train = Dataset.from_pandas(train_df)
    # dataset_test = Dataset.from_pandas(test_df)

    # Combine them into a single DatasetDict
    dataset = DatasetDict({
        'train': dataset_train,
        # 'test': dataset_test
    })

    print("\nDataset Format: ", dataset)

    ### Define Class Weight ###
    class_weights = (1 / train_df['label'].value_counts(normalize=True).sort_index()).tolist()
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()
    print(f"\n\nClass Weight: {class_weights}")

    ### Preprocess Data ###
    def data_preprocesing(row):
        return tokenizer(row['text'], padding=True, truncation=True, max_length=max_length)

    tokenized_data = dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
    tokenized_data.set_format("torch")

    print('-' * 50)
    print("Tokenizing Data Format")
    print(tokenized_data['train'])
    print('-' * 50)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    wandb.init(
        project="solution_loc_llm",
    )

    print(f"Model Name: {model_name}")
    print(f"Batch Size: {BS}, Learning Rate: {LR}, Epochs: {E}, Oversample: {oversample}")
    print(f"Class Weights: {class_weights}")
    print(f"Max Length: {max_length}, Token Batch Size: {token_batch_size}")
    print(f"Logging Steps: {logging_steps}, Weight Decay: {weight_decay}")
    print(f"Save Location: {model_to_save_path}")

    training_args = TrainingArguments(
        output_dir='output',
        learning_rate=LR,
        per_device_train_batch_size=BS,
        # per_device_eval_batch_size=bs,
        num_train_epochs=E,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        # eval_strategy='epoch',
        save_strategy='epoch',
        # load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        # eval_dataset=tokenized_data['test'],
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    if model_to_save_path is not None:
        if not os.path.exists(model_to_save_path):
            os.makedirs(model_to_save_path)

        print("Saving Model ...")

        trainer.model.save_pretrained(model_to_save_path)
        # model.save_pretrained(save_loc)
        tokenizer.save_pretrained(model_to_save_path)

        print(f"Model saved to {model_to_save_path}")

def prepare_dataset(data):
    issue_ids = data['issue_id'].unique()
    # print(f"Unique issue IDs: {len(issue_ids)}")
    # print(issue_ids)
    splits = []
    for i in range(len(issue_ids)):
        test_ids = [issue_ids[i]]
        train_ids = [id_ for j, id_ in enumerate(issue_ids) if j != i]
        splits.append({'train': train_ids, 'test': test_ids})
    # # Example: print the splits
    # for idx, split in enumerate(splits):
    #     print(f"Split {idx+1}:")
    #     print(f"  Train IDs: {split['train']}")
    #     print(f"  Test ID: {split['test']}")
    return splits


def load_base_model(model_name):
    set_llm_seed(42)

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
        device_map='sequential',  # 'auto' for automatic device mapping
        cache_dir="/scratch/mehedi/models"
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

    ### Get Tokenizer ###
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


def load_finetuned_model(finetuned_path):
    model_name = "meta-llama/Meta-Llama-3-8B"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

    # TODO Check if both base model produce same result

    # Load the base model
    # base_model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name,
    #     device_map='auto',
    #     torch_dtype=torch.bfloat16,
    #     load_in_4bit=True
    # )

    base_model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, finetuned_path) # TODO to check if it works
    model = model.merge_and_unload()  # Merge LoRA weights into the base model

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


if __name__ == "__main__":
    # Best ML model
    model_name = "meta-llama/Meta-Llama-3-8B"
    fine_tuned_model_path = f"generalizability/models/plm/roberta-base/mozilla/roberta-base-ft-mozilla"
    # Best configuration for roberta-base of Mozilla experiments
    oversample = False
    BS = 16
    LR = 3.00E-05
    E = 10

    # We experiment with the datasets of two new projects
    projects = ['Gnucash', 'Chromium']
    # projects = ['Chromium']

    # On these two datasets, we evaluate the performance of three trained models:
        # 1. mozilla: Model trained on the Mozilla data
        # 2. project: Model trained on the project (i.e., GunCash or Chromium) data
        # 3. mozilla-project: Model trained on both Mozilla and the project (i.e., GunCash or Chromium) data
    # exp_types = ['mozilla', 'project', 'mozilla-project']
    exp_types = ['mozilla-project']

    for project in projects:
        for exp_type in exp_types:
            if exp_type == 'mozilla':
                exp_name = "RoBERTa-FT-Mozilla"
            elif exp_type == 'project':
                exp_name = f"RoBERTa-FT-{project}"
            elif exp_type == 'mozilla-project':
                exp_name = f"RoBERTa-FT-Mozilla-{project}"

            data_path = f"generalizability/dataset/{project}.csv"
            model_to_save_path = f"generalizability/models/plm/{model_name}/{project}/{exp_name}"
            prediction_path = f"generalizability/predictions/plm/{model_name}/{project}/{exp_name}_predictions.csv"
            result_path = f"generalizability/results/plm/{model_name}/{project}/{exp_name}_results.csv"

            project_data = pd.read_csv(data_path)
            X_test_project = project_data['text'].tolist()
            y_test_project = project_data['label'].tolist()

            y_vals = []
            y_preds = []
            result_df = pd.DataFrame(columns=['Oversample', 'Batch_Size', 'Epochs', 'LR', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TP', 'FP', 'TN', 'FN'])
            prediction_df = pd.DataFrame()

            # Load the fine-tuned model on Mozilla dataset
            model, tokenizer = load_finetuned_model(model_to_save_path)

            # Run experiment 1
            if exp_type == 'mozilla':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                y_pred = get_predictions(X_test_project, model, tokenizer)
                y_preds.extend(y_pred)
                y_vals.extend(y_test_project)

                project_data[exp_name] = y_pred
                prediction_df = pd.concat([prediction_df, project_data], ignore_index=True)

            # Run experiment 2 or 3
            elif exp_type == 'project' or exp_type == 'mozilla-project':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                folds = prepare_dataset(project_data)
                for i, fold in enumerate(folds):
                    print(f"Fold {i + 1}/{len(folds)}")

                    model_to_save_path_new = f"{model_to_save_path}/{i}"

                    train_issue_id = fold['train']
                    test_issue_id = fold['test']
                    train_df = project_data[project_data['issue_id'].isin(train_issue_id)]
                    test_df = project_data[project_data['issue_id'].isin(test_issue_id)]

                    train_df = train_df[['text', 'label']]

                    X_test = test_df['text'].tolist()
                    y_test = test_df['label'].tolist()

                    # Load the base model
                    if exp_type == 'project':
                        model, tokenizer = load_base_model(model_name)

                    fine_tune_model(train_df, model, tokenizer, oversample, BS, LR, E, model_to_save_path_new)

                    # Load the new fine-tuned model
                    model, tokenizer = load_finetuned_model(model_to_save_path)
                    y_pred = get_predictions(X_test, model, tokenizer)

                    y_vals.extend(y_test)
                    y_preds.extend(y_pred)

                    test_df = test_df.copy()
                    test_df[exp_name] = y_pred

                    prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

            FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
            result = [oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN]
            result_df = pd.concat([result_df, pd.DataFrame([result], columns=result_df.columns)], ignore_index=True)

            print("=" * 50)
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print("=" * 50)

            # Save results
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_df.to_csv(result_path, index=False)
            # Save predictions
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            prediction_df.to_csv(prediction_path, index=False)
            print(f"Predictions saved to: {prediction_path}")

