import ast
import os

from generalizability.plm_experiments_antu import prepare_dataset
from sources.b_preprocess_data import text_preprocess
from sources.utils import *

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
# from b_preprocess_data import text_preprocess
# import wandb

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
import joblib


def oversampling_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Train data distribution After Oversampling: {y_train_resampled.value_counts()}")

    return X_train_resampled, y_train_resampled


def get_embedding(X_test, X_train, mode):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[f'text_{mode}'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[f'text_{mode}'])

    # Converting sparse matrices to dense
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    return X_test_dense, X_train_dense

def get_lm_embedding(data, embedder):
    train_embeddings = data[embedder + '_embedding'].apply(ast.literal_eval).tolist()
    train_embeddings = np.array(train_embeddings)
    return train_embeddings

def get_splitted_data(train_df, test_df, mode, embedding, oversampling):
    solution_col = 'label'
    X_train, y_train = train_df.drop(solution_col, axis=1), train_df[solution_col]
    X_test, y_test = test_df.drop(solution_col, axis=1), test_df[solution_col]

    if embedding in ["bert", "llama", "gpt"]:
        X_test_embedding = get_lm_embedding(X_test, embedding)
        X_train_embedding = get_lm_embedding(X_train, embedding)
    else:
        X_test_embedding, X_train_embedding = get_embedding(X_test, X_train, mode)

    if oversampling:
        X_train_embedding, y_train = oversampling_data(X_train_embedding, y_train)

    return X_train_embedding, X_test_embedding, y_train, y_test


def prepare_data_for_ml():
    exp_config = get_experiment_config()
    runs = exp_config.get("eval_run", [])

    for run in runs:
        dev_df = pd.read_csv(dev_data_path.replace('<run>', f'{run}'))
        test_df = pd.read_csv(test_data_path.replace('<run>', f'{run}'))
        train_df = pd.read_csv(train_data_path.replace('<run>', f'{run}'))
        val_df = pd.read_csv(val_data_path.replace('<run>', f'{run}'))

        dev_df['text'].fillna('', inplace=True)
        test_df['text'].fillna('', inplace=True)
        train_df['text'].fillna('', inplace=True)
        val_df['text'].fillna('', inplace=True)

        print(f"Dev DF Path: {dev_data_path.replace('<run>', f'{run}')}\nDev columns: {dev_df.columns}")
        print(f"Train DF Path: {train_data_path.replace('<run>', f'{run}')}\nTrain columns: {train_df.columns}")

        for mode in exp_config['preprocess']:
            for embedding in exp_config['embedding']:
                if embedding in ["bert", "llama", "gpt"] and mode in ["essential", "all"]:
                    continue

                for oversampling in exp_config['oversampling']:
                    print(f"Mode: {mode}, Embedding: {embedding}, Oversampling: {oversampling}")

                    print("\nAdding preprocessing ... ...")
                    dev_df[f'text_{mode}'] = dev_df['text'].apply(text_preprocess, mode=mode)
                    test_df[f'text_{mode}'] = test_df['text'].apply(text_preprocess, mode=mode)
                    train_df[f'text_{mode}'] = train_df['text'].apply(text_preprocess, mode=mode)
                    val_df[f'text_{mode}'] = val_df['text'].apply(text_preprocess, mode=mode)

                    X_dev_embedding, X_test_embedding, y_dev, y_test = get_splitted_data(dev_df, test_df, mode, embedding, oversampling)
                    X_train_embedding, X_val_embedding, y_train, y_val = get_splitted_data(train_df, val_df, mode, embedding, oversampling)

                    data_dict = {
                        'X_dev_embedding': X_dev_embedding,
                        'X_test_embedding': X_test_embedding,
                        'y_dev': y_dev,
                        'y_test': y_test,
                        'X_train_embedding': X_train_embedding,
                        'X_val_embedding': X_val_embedding,
                        'y_train': y_train,
                        'y_val': y_val
                    }

                    path = f"{dataset_fold_path}/{run}"
                    filename = f'data__preprocess_{mode}__embedding_{embedding}__oversampling_{"os" if oversampling else "nos"}.joblib'
                    full_path = os.path.join(path, filename)
                    joblib.dump(data_dict, full_path)

                    print(f"Data saved to {full_path} for run {run}, mode {mode}, embedding {embedding}, oversampling {oversampling}")

def set_ml_seed(RANDOM_STATE):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

def getBaseModel(clf_name):
    if clf_name == "Logistic_Regression":
        return LogisticRegression(max_iter=10000, random_state=42)
    elif clf_name == "Naive_Bayes":
        return GaussianNB()
    elif clf_name == "K_Nearest_Neighbors":
        return KNeighborsClassifier()
    elif clf_name == "Decision_Tree":
        return DecisionTreeClassifier(random_state=42)
    elif clf_name == "SVC":
        return SVC(random_state=42)
    elif clf_name == "Random_Forest":
        return RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown classifier name: {clf_name}")
    
def get_tuned_model_prediction(X_test, X_train, clf, model, config, y_train, model_save_path):
    if not config:
        searchCV = GridSearchCV(clf,
                            param_grid=model["params"],
                            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                            scoring='f1',
                            n_jobs=-1,
                            verbose=3)
    
        searchCV.fit(X_train, y_train)
        
        print("Best Parameters:", searchCV.best_params_)
        
        best_model = searchCV.best_estimator_
        config = searchCV.best_params_
        
        print("Best Cross-Validation Score:", searchCV.best_score_)
        
    else:
        best_model = clf.set_params(**config)
        
        best_model.fit(X_train, y_train)
        
    y_pred = best_model.predict(X_test)
    
    save_model(best_model, model_save_path)
    return config, y_pred

def ml_model(data_dict, result_path, model_base_path, run, mode, embedding, oversampling, model, config):
    X_dev_embedding = data_dict['X_dev_embedding']
    X_test_embedding = data_dict['X_test_embedding']
    y_dev = data_dict['y_dev']
    y_test = data_dict['y_test']

    set_ml_seed(42)
    
    clf = getBaseModel(model["model_name"])

    model_save_path = f'model_{model["model_name"]}_preprocess_{mode}__embedding_{embedding}__oversampling_{"os" if oversampling else "nos"}.joblib'
    model_path = os.path.join(model_base_path, model_save_path)

    param, y_pred = get_tuned_model_prediction(X_test_embedding, X_dev_embedding, clf, model, config, y_dev, model_path)

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)

    result = [mode, embedding, oversampling, model['model_name'], run, accuracy, precision, recall, f1,
              TP, FP, TN, FN, param]

    print("=" * 50)
    print(f"Preprocessing Mode: {mode}")
    print(f"Embedding: {embedding}")
    print(f"Oversampling: {oversampling}")
    print(f"Model name: {model['model_name']}")
    print(f"Run: {run}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print(f"Parameters: {param}")
    print("=" * 50)

    append_row_to_csv(result_path, result)
    print(f"Results are saved to {result_path}")

    return y_pred

def process_config(mode, embedding, oversampling, model, config, path, run, result_path, model_base_path, test_predictions):
    if embedding in ["bert", "llama", "gpt"] and mode in ["essential", "all"]:
        return None

    try:
        print(f"Preprocess: {mode}, Embedding: {embedding}, Oversampling: {oversampling}, Model: {model['model_name']}, Run: {run}")
        data_dict = joblib.load(os.path.join(os.path.dirname(path), f'data__preprocess_{mode}__embedding_{embedding}__oversampling_{"os" if oversampling else "nos"}.joblib'))
        y_pred = ml_model(data_dict, result_path, model_base_path, run, mode, embedding, oversampling, model, config)
        column_name = '_'.join(
            [mode, embedding, "os" if oversampling else "nos", model['model_name'],
             str(run)])
        test_predictions[column_name] = y_pred
        print("Prediction Loaded to Test Prediction File")

    except Exception as e:
        print(f"An error occurred: {e}")
    return test_predictions


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


def ml_run(run_best_config=False):
    exp_config = get_experiment_config()
    runs = exp_config.get("eval_run", [])

    result_path = "results/ml/comment_data_results.csv"
    model_base_path = "models/ml"
    print(f"Result Path: {result_path}")

    result_header = ["preprocess", "embedding", "oversampling", "model", "run", "accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN", "param"]
    append_row_to_csv(result_path, result_header)
    print(f"{result_header} is saved to {result_path}")

    for run in runs:
        path = f"dataset/solution_identification_data/folds/{run}/comment_test_data.csv"
        test_df = pd.read_csv(path)
        test_predictions = test_df[['issue_id', 'text_id', 'text', 'code', 'label']]

        if run_best_config:
            tasks = []
            for model in exp_config['models']:

                if model['model_name'] != "SVC": # TODO Remove condition if necessary
                    continue

                model_details = exp_config['best_model_config'][str(run)][model['model_name']]
                preprocess = model_details['preprocess']
                embedding = model_details['embedding']
                oversampling = model_details['oversampling']
                config = model_details['config']
                tasks.append((preprocess, embedding, oversampling, model, config, path, run, result_path, model_base_path, test_predictions.copy()))
        else:
            tasks = [
                (mode, embedding, oversampling, model, None, path, run, result_path, model_base_path, test_predictions.copy())
                for model in exp_config['models']
                for oversampling in exp_config['oversampling']
                for embedding in exp_config['embedding']
                for mode in exp_config['preprocess']
                if not (embedding in ["bert", "llama", "gpt"] and mode in ["essential", "all"])
            ]

        for task in tasks:
            result = process_config(*task)
            if result is not None:
                test_predictions = test_predictions.merge(result, how='left')

                os.makedirs(ml_prediction_comment_dir, exist_ok=True)
                test_predictions.to_csv(os.path.join(ml_prediction_comment_dir, f"run_{run}.csv"), index=False)
                print(f"Test Prediction Saved to {path}")


if __name__ == "__main__":

    projects = ['chromium', 'gnucash']

    for project in projects:
        projects_embedding_data_path = f"generalizability/dataset/{project}_with_embeddings.csv"

        data = pd.read_csv(projects_embedding_data_path)

        folds = prepare_dataset(data)
        for i, fold in enumerate(folds):
            prepare_data_for_ml()
            ml_run(True)
