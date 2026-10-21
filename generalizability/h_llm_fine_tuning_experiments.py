import wandb
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F

from utils import *
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


def train_evaluate_llm(model_name, train_df, oversample, bs, lr, e, save_loc=None):
    set_llm_seed(42)

    ### Innitial Params ###
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
        device_map='sequential', # 'auto' for automatic device mapping
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

    wandb.init(
        project="solution_loc_llm",
    )

    print(f"Model Name: {model_name}")
    print(f"Batch Size: {bs}, Learning Rate: {lr}, Epochs: {e}, Oversample: {oversample}")
    print(f"Class Weights: {class_weights}")
    print(f"Max Length: {max_length}, Token Batch Size: {token_batch_size}")
    print(f"Logging Steps: {logging_steps}, Weight Decay: {weight_decay}")
    print(f"Save Location: {save_loc}")

    training_args = TrainingArguments(
        output_dir='output',
        learning_rate=lr,
        per_device_train_batch_size=bs,
        # per_device_eval_batch_size=bs,
        num_train_epochs=e,
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

    separator_print("TRAIN DATASET")
    print(tokenized_data['train'])

    # train_result = trainer.train()
    trainer.train()

    # separator_print("TRAIN RESULT")
    # print(train_result)

    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        separator_print("Saving Model ...")
        
        trainer.model.save_pretrained(save_loc)
        # model.save_pretrained(save_loc)
        tokenizer.save_pretrained(save_loc)

        print(f"Model saved to {save_loc}")

    # ### Evaluation ###
    # def generate_predictions(model, df_test, bs, max_length):
    #     sentences = df_test['text'].tolist()
    #     batch_size = bs
    #     all_outputs = []

    #     for i in range(0, len(sentences), batch_size):
    #         batch_sentences = sentences[i:i + batch_size]

    #         inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True,
    #                            max_length=max_length)

    #         inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu')
    #                   for k, v in inputs.items()}

    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             all_outputs.append(outputs['logits'])

    #     final_outputs = torch.cat(all_outputs, dim=0)
    #     return final_outputs.argmax(axis=1).cpu().numpy()
    
    # model.eval()

    # y_pred = generate_predictions(model, test_df, bs, max_length)
    # return y_pred


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
    

def predict_with_peft_model(sentences, adapter_path, base_model_name, max_length=1024, batch_size=8):
    set_llm_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Quantization config (must match training)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # 1. Load the base model (with quantization)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        num_labels=2,
        device_map='sequential'
    )
    # 2. Load the LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 3. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    all_outputs = []
    model.eval()
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


if __name__ == "__main__":
    print_current_time()

    # Define the model and data paths
    result_path = f"test_llm/results/llm_fine_tuning/comment_data_results.csv"
    prediction_path = f"test_llm/results/predictions/llm/comment_data_predictions.csv"
    model_folder = f"test_llm/models/llm/"

    result = ["oversample", "batch_size", "epoch", "initial_learning_Rate", "model_name", "accuracy",
                "precision", "recall", "f1", "TP", "FP", "TN", "FN"]
    append_row_to_csv(result_path, result)

    # Define the model configurations
    exp_config = get_experiment_config()
    model_name = exp_config['llm_model_conf']['model_name']
    BS = exp_config['llm_model_conf']['params']['batch_size']
    E = exp_config['llm_model_conf']['params']['epoch']
    LR = exp_config['llm_model_conf']['params']['learning_rate']
    oversample = exp_config['llm_model_conf']['params']['oversample']

    train_df = pd.read_csv(mapped_comment_data)

    if oversample:
        print("Oversampling the training data ...")
        train_df = balance_classes(train_df, 'label', output_type='df')

    train_evaluate_llm(model_name, train_df, oversample, BS, LR, E, model_folder)
    # FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, test_df['label'].tolist())

    # print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    # result_ft = [
    #     oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN
    # ]
    # append_row_to_csv(result_path, result_ft)

    # test_df['predictions'] = y_pred

    # os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    # test_df.to_csv(prediction_path, index=False)

    # print(f"Predictions saved to {prediction_path}")

    # # Testing again with saved model
    # print("Testing with saved model completed.")
    # y_pred = predict_with_peft_model(
    #     sentences=test_df['text'].tolist(),
    #     adapter_path=model_folder,
    #     base_model_name=model_name,
    #     max_length=1024,
    #     batch_size=BS
    # )
    # FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, test_df['label'].tolist())
    # print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")

    # test_df['saved_model_predictions'] = y_pred

    # saved_prediction_path = f"test_llm/results/predictions/llm/saved_model_predictions.csv"
    # test_df.to_csv(saved_prediction_path, index=False)

    print_current_time()