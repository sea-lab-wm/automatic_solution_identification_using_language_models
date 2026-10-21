import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import BertTokenizer, BertModel
import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.embeddings.ollama import OllamaEmbedding   # pip install llama-index-embeddings-ollama
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# For OpenAI API key error, run: source ~/.bash_profile
# Load the tokenizer for GPT-J (as a proxy for GPT-4 compatibility)
gpt_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
client = OpenAI()


# Get BERT embeddings by Hugging Face's Transformers
def get_bert_embeddings(text):
    # Load BERT model and tokenizer
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Encode the text into tokens with attention masks
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # Move encoded input to the same device as model
    encoded_input = {key: val.to(bert_model.device) for key, val in encoded_input.items()}

    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**encoded_input)

    # Use the pooled output as a simple sentence-level embedding
    embeddings = outputs.pooler_output
    return embeddings.cpu().numpy()  # Ensure numpy array is on CPU


# Get GPT embeddings by OpenAI's API
def get_gpt_embeddings(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")

    # Tokenize the input text and truncate if necessary
    tokens = gpt_tokenizer.tokenize(text)
    if len(tokens) > 8000:
        # Convert only the first 8000 tokens back to a string
        text = gpt_tokenizer.convert_tokens_to_string(tokens[:8000])

    # Get the embedding using the potentially truncated text
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Get LLAMA embeddings by OLLAMA
def get_llama_embeddings(text):
    ollama_embedding = OllamaEmbedding(model_name="llama3", base_url="http://localhost:11434", ollama_additional_kwargs={"mirostat": 0})

    text = text.replace("\n", " ")
    # Tokenize the input text and truncate if necessary
    tokens = gpt_tokenizer.tokenize(text)
    if len(tokens) > 2000:
        # Convert only the first 8000 tokens back to a string
        text = gpt_tokenizer.convert_tokens_to_string(tokens[:2000])
    embeddings = ollama_embedding.get_query_embedding(text)
    # print(embeddings)
    return embeddings


def count_tokens(text):
    tokens = gpt_tokenizer.tokenize(text)
    return len(tokens)


if __name__ == '__main__':
    embedders = ['bert', 'gpt', 'llama']

    dataset = 'chromium'    # "chromium" or "gnucash"

    # dataset_path = '../dataset/solution_identification_data/labeled_comment_data.csv'
    # embedding_path = '../dataset/solution_identification_data/labeled_comment_data_with_embeddings.csv'

    dataset_path = f'../generalizability/dataset/{dataset}.csv'
    embedding_path = f'../generalizability/dataset/{dataset}_with_embeddings.csv'

    dataset_df = pd.read_csv(dataset_path)

    for embedder in embedders:
        if embedder == 'bert':
            # Fetch BERT embeddings and store them as NumPy arrays
            embeddings = np.array([get_bert_embeddings(str(text))[0] for text in tqdm(dataset_df['text'], desc='Generating BERT embeddings')])

            # Add new columns to the dataframes
            dataset_df['bert_embedding'] = embeddings.tolist()
            print("BERT embeddings fetched!\n")
        elif embedder == 'gpt':
            # Fetch GPT embeddings and store them as NumPy arrays
            embeddings = np.array([get_gpt_embeddings(str(text)) for text in tqdm(dataset_df['text'], desc='Generating GPT embeddings')])

            # Add new columns to the dataframes
            dataset_df['gpt_embedding'] = embeddings.tolist()
            print("GPT embeddings fetched!\n")
        elif embedder == 'llama':
            # Fetch LLAMA embeddings and store them as NumPy arrays
            embeddings = np.array([get_llama_embeddings(str(text)) for text in tqdm(dataset_df['text'], desc='Generating LLAMA embeddings')])

            # Add new columns to the dataframes
            dataset_df['llama_embedding'] = embeddings.tolist()
            print("LLAMA embeddings fetched!\n")

    # Save the updated dataframes to CSV files
    dataset_df.to_csv(embedding_path, index=False)
    print("Dataframes saved to CSV files!")
