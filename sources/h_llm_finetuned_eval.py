import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from utils import calculate_results
torch.cuda.empty_cache()

# Specify the path to your saved model and tokenizer
model_path = "models/llm/0/meta-llama/Meta-Llama-3-8B__BS_8__E_5__LR_1e-05"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1

# Move model to appropriate device
device = torch.device("cuda")
model.to(device)
model.eval()

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

test_df = pd.read_csv("dataset/new_data/chromium.csv")
y_pred = generate_predictions(model, test_df, 8, 1024)

test_df['llm_pred'] = y_pred

FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, test_df['label'])

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

test_df.to_csv('test_llama_pred.csv', index=False)