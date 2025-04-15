from utils import *

import numpy as np
import re
import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available()) 
import torch.nn.functional as F

from datasets import DatasetDict, Dataset
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, classification_report
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


def train_evaluate_llm(model_name, train_df, test_df, bs, lr, e, save_loc=None):
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
    dataset_train = Dataset.from_pandas(train_df)
    dataset_test = Dataset.from_pandas(test_df)

    # Combine them into a single DatasetDict
    dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    print("\nDataset Format: ", dataset)

    ### Define Class Weight ###
    class_weights = (1 / train_df[target_column].value_counts(normalize=True).sort_index()).tolist()
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()
    print(f"\n\nClass Weight: {class_weights}")

    ### Model Load ###
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

    ### Preprocess Data ###
    def data_preprocesing(row):
        return tokenizer(row['text'], padding=True, truncation=True, max_length=max_length)

    tokenized_data = dataset.map(data_preprocesing, batched=True, remove_columns=['text'])
    tokenized_data.set_format("torch")

    separator_print("Tokenized Data")
    print(tokenized_data)

    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='output',
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=e,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    separator_print("TRAIN DATASET")
    print(tokenized_data['train'])

    train_result = trainer.train()

    separator_print("TRAIN RESULT")
    print(train_result)

    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        
        model.save_pretrained(save_loc)
        tokenizer.save_pretrained(save_loc)

    ### Evaluation ###
    def generate_predictions(model, df_test, bs, max_length):
        sentences = df_test['text'].tolist()
        batch_size = bs
        all_outputs = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]

            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True,
                               max_length=max_length)

            inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu')
                      for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                all_outputs.append(outputs['logits'])

        final_outputs = torch.cat(all_outputs, dim=0)
        return final_outputs.argmax(axis=1).cpu().numpy()

    y_pred = generate_predictions(model, test_df, bs, max_length)
    return y_pred


def llama_experiments(run, mod, result_path, prediction_path, oversample, hyperparam_tuning=False):
    exp_config = get_experiment_config()
    
    ### Data Retrieve ###
    dev_df = pd.read_csv(dev_data_path.replace('<run>', f'{run}'))
    test_df = pd.read_csv(test_data_path.replace('<run>', f'{run}'))
    train_df = pd.read_csv(train_data_path.replace('<run>', f'{run}'))
    val_df = pd.read_csv(val_data_path.replace('<run>', f'{run}'))
    train_df_os = pd.read_csv(train_os_data_path.replace('<run>', f'{run}'))
    dev_df_os = pd.read_csv(dev_os_df_path.replace('<run>', f'{run}'))

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

    dev_df = dev_df[['text', 'label']]
    test_df = test_df[['text', 'label']]
    train_df = train_df[['text', 'label']]
    val_df = val_df[['text', 'label']]

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

    model_save_path = f"models/llm/{run}/{model_name.replace(' ', '_')}__BS_{BS}__E_{E}__LR_{LR}"
    y_pred = train_evaluate_llm(model_name, dev_df, test_df, BS, LR, E, save_loc=model_save_path)
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
    print(f"\n\Results succesfully saved to {result_path}\n\n")

    os.makedirs(prediction_path, exist_ok=True)

    test_df_copy = test_df.copy()
    test_df_copy[model_name + f"_{'os' if oversample else 'nos'}_prediction"] = y_pred

    path = os.path.join(prediction_path, f"model_name_{'os' if oversample else 'nos'}_predictions.csv")
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
                y_pred = train_evaluate_llm(model_name, train_df, val_df, bs, lr, e)
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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()

        outputs = model(**inputs)

        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    exp_config = get_experiment_config()
    prepare_data(mapped_comment_data, train_test_comment_data)

    target_column = 'label'

    res_path = "results/llm_fine_tuning/comment_data_results.csv"
    result = ["run", "oversample", "batch_size", "epoch", "initial_learning_Rate", "model_name", "accuracy",
                "precision", "recall", "f1", "TP", "FP", "TN", "FN"]
    
    runs = 1
    append_row_to_csv(res_path, result)

    for run in range(runs):
        for model in exp_config['llm_model']:
            pred_path = comment_data_llama_prediction.replace("<run>", f'{run}')
            oversampling = True
            try:
                llama_experiments(run, model, res_path, pred_path, oversample=oversampling, hyperparam_tuning=False)
            except Exception as exception:
                print(f"\n\nAn error occurred: {exception}\n\n")
                print('*' * 50)
                print(f"Model Name ::: {model['model_name']}")
                print(f"Oversampling: {oversampling}")
                print(f"Run ::: {run}")
                print('*' * 50)