import ast
import json
import os
import joblib
import numpy as np

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

from b_preprocess_data import text_preprocess
from d_mlm_experiments import getBaseModel
from utils import calculate_results

def get_experiment_config():
    experiment_file = 'experiments.json'
    with open(experiment_file, 'r') as file:
        experiments = json.load(file)
    return experiments

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def get_embedding(X_test, config):
    mode = config['preprocess']
    run = config['run']

    dev_df = pd.read_csv(f"dataset/solution_identification_data/folds/{run}/comment_dev_data.csv")

    dev_df[f'text_{mode}'] = dev_df['text'].apply(text_preprocess, mode=mode)
    X_test[f'text_{mode}'] = X_test['text'].apply(text_preprocess, mode=mode)

    texts = pd.concat([dev_df[f'text_{mode}'], X_test[f'text_{mode}']], ignore_index=True)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(texts)

    X_train_tfidf = tfidf_vectorizer.transform(dev_df[f'text_{mode}'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[f'text_{mode}'])

    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    return X_test_dense, X_train_dense, dev_df['label']

def get_lm_embedding(data, embedder):
    train_embeddings = data[embedder + '_embedding'].apply(ast.literal_eval).tolist()
    train_embeddings = np.array(train_embeddings)
    return train_embeddings

def oversampling_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Train data distribution After Oversampling: {y_train_resampled.value_counts()}")

    return X_train_resampled, y_train_resampled

def eval(data, model_path, config):
    solution_col = 'label'

    X_test, y_test = data.drop(solution_col, axis=1), data[solution_col]
    if config['embedding'] in ["bert", "llama", "gpt"]:
        X_test_embedding = get_lm_embedding(X_test, config['embedding'])
        best_model = load_model(model_path)
    else:
        X_test_embedding, X_train, y_train = get_embedding(X_test, config)
        exp_config = get_experiment_config()
        clf = getBaseModel(config["model"])
        model_config = exp_config['best_model_config'][str(config['run'])][config['model']]['config']
        best_model = clf.set_params(**model_config)
        best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test_embedding)
    data[f'prediction__run_{config["run"]}__model_{config["model"]}'] = y_pred

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)

    print("=" * 50)
    print(f"Preprocessing Mode: {config['preprocess']}")
    print(f"Embedding: {config['embedding']}")
    print(f"Oversampling: {config['oversampling']}")
    print(f"Model name: {config['model']}")
    print(f"Run: {config['run']}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print("=" * 50)

    output = f"results/ml/{config['run']}/eval_data_results.csv"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Saved to {output}")

    return data,accuracy,precision,recall,f1,TP,FP,TN,FN

if __name__ == "__main__":
    eval_dataset = "dataset/new_data/gnucash_data_with_embeddings.csv"
    ml_model_base_path = "models/ml/<run>"
    eval_results = 'results/ml/eval_data_results.csv'
    runs = 1

    results = pd.DataFrame([], columns=['preprocess','embedding','oversampling','model','run','accuracy','precision','recall','f1','TP','FP','TN','FN'])
    for run in range(runs):
        folder_path = ml_model_base_path.replace("<run>", str(run))
        models_in_run = {}
        data = pd.read_csv(eval_dataset)
        for filename in os.listdir(folder_path):
            if filename.endswith(".joblib"):
                model_name = filename.replace(".joblib", "")
                full_path = os.path.join(folder_path, filename)
                splits = model_name.split('_')
                config = {
                    'model': '_'.join(splits[splits.index('model')+1: splits.index('preprocess')]),
                    'preprocess': splits[splits.index('preprocess')+1],
                    'embedding': splits[splits.index('embedding')+1],
                    'oversampling': splits[splits.index('oversampling')+1],
                    'run': run
                }
                data,accuracy,precision,recall,f1,TP,FP,TN,FN = eval(data, full_path, config)

                results.loc[len(results)] = {
                    'preprocess':config['preprocess'],
                    'embedding':config['embedding'],
                    'oversampling':config['oversampling'],
                    'model':config['model'],
                    'run':config['run'],
                    'accuracy':accuracy,
                    'precision':precision,
                    'recall':recall,
                    'f1':f1,
                    'TP':TP,
                    'FP':FP,
                    'TN':TN,
                    'FN':FN
                }
        
        output = f"results/ml/{run}/eval_data_results.csv"
        data.drop(['bert_embedding', 'gpt_embedding', 'llama_embedding'], axis=1, inplace=True)
        data.to_csv(output, index=False)
        print(f"Saved to {output}")

    results.to_csv(eval_results, index=False)
    print(f"Saved to {eval_results}")
    