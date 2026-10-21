import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

from utils import calculate_results

def predict(texts, model, tokenizer, max_length=1024, batch_size=8):
    all_outputs = []
    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_outputs.extend(predictions.cpu().numpy())

    return all_outputs

save_loc = "models/llm/0/meta-llama/Meta-Llama-3-8B__BS_8__E_5__LR_1e-05"
model_name = "meta-llama/Meta-Llama-3-8B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_loc)

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map='sequential',
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, save_loc)
model = model.merge_and_unload()  # Merge LoRA weights into the base model

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

model.eval()

# Example usage
test_df = pd.read_csv("results/predictions/llm/folds/0/llm_eval_predictions.csv")

texts = test_df['text'].tolist()
predictions = predict(texts, model, tokenizer)

test_df['predicted_label'] = predictions

FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(predictions, test_df['label'])

print("=" * 50, "Fine tuned", "=" * 50)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print("=" * 50)

test_df.to_csv('results/predictions/llm/folds/0/llm_eval_predictions.csv', index=False)