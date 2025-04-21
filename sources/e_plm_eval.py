import json

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import tensorflow as tf

import ktrain
import numpy as np

import pandas as pd

from utils import calculate_results

def set_seed(seed):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    print(f"Random seed set as {seed}")

def get_experiment_config():
    experiment_file = 'experiments.json'
    with open(experiment_file, 'r') as file:
        experiments = json.load(file)
    return experiments

def load_model(filename):
    predictor = ktrain.load_predictor(filename)
    model = ktrain.get_predictor(predictor.model, predictor.preproc)
    print(f"Model loaded from {filename}")
    return model

def eval(data, model_path, config):
    set_seed(42)
    exp_config = get_experiment_config()
    solution_col = 'label'

    X_test, y_test = data.drop(solution_col, axis=1), data[solution_col]

    predictor = ktrain.load_predictor(model_path)
    best_model = ktrain.get_predictor(predictor.model, predictor.preproc)

    y_pred = best_model.predict(X_test['text'].tolist())
    data[f'prediction__run_{config["run"]}__model_{config["model"]}'] = y_pred

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)

    model_name = config['model']
    run = config['run']
    model_config = exp_config['best_model_config'][str(run)][model_name]

    print("=" * 50)
    print(f"batch_sizes: {model_config['batch_size']}")
    print(f"total_epochs: {model_config['epoch']}")
    print(f"learning_rate: {model_config['learning_Rate']}")
    print(f"Model name: {model_name}")
    print(f"Oversampling: {model_config['oversample']}")
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

    output = f"results/plm/{run}/eval_data_results.csv"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    data.to_csv(output, index=False)
    print(f"Saved to {output}")

    return data,accuracy,precision,recall,f1,TP,FP,TN,FN

if __name__ == "__main__":
    eval_dataset = "dataset/new_data/chromium.csv"
    plm_model_base_path = "models/plm/<run>"
    eval_results = 'results/plm/eval_data_results.csv'
    runs = 1
    exp_config = get_experiment_config()

    results = pd.DataFrame([], columns=['run','oversample','batch_size','epoch','learning_Rate','model_name','accuracy','precision','recall','f1','TP','FP','TN','FN'])
    for run in range(runs):
        folder_path = plm_model_base_path.replace("<run>", str(run))
        models_in_run = {}
        data = pd.read_csv(eval_dataset)
        for filename in os.listdir(folder_path):
            if filename.endswith(".joblib"):
                model_name = filename.replace(".joblib", "")
                full_path = os.path.join(folder_path, filename)
                splits = model_name.split('__')
                config = {
                    'model': splits[0],
                    'run': run
                }
                
                data,accuracy,precision,recall,f1,TP,FP,TN,FN = eval(data, full_path, config)

                model_name = config['model']
                model_config = exp_config['best_model_config'][str(run)][model_name]

                results.loc[len(results)] = {
                    'run': run,
                    'oversample': model_config['oversample'],
                    'batch_size': model_config['batch_size'],
                    'epoch': model_config['epoch'],
                    'learning_Rate': model_config['learning_Rate'],
                    'model_name': model_name,
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
    