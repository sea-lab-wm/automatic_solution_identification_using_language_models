import ast

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
    
def get_tuned_model_prediction(X_test, X_train, clf, config, y_train, model_save_path):
    if not config:
        searchCV = GridSearchCV(clf,
                            param_grid=config,
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

def ml_model(X_train , y_train, X_test, y_test, model_to_save_path_new,fine_tuned_model_path, run, mode, embedding, oversampling, model_name, config):

    set_ml_seed(42)

    if model_type == 'base':
        clf = getBaseModel(model_name)
    else:
        clf = load_model(fine_tuned_model_path)

    _, y_pred = get_tuned_model_prediction(X_test, X_train, clf, config, y_train, model_to_save_path_new)

    return y_pred


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


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

if __name__ == "__main__":

    dataset = 'chromium'    # "chromium" or "gnucash"
    model_name = "SVC"
    model_type = 'base'  # "base" or "fine-tuned"
    run = '0'

    if model_type == 'base':
        exp_name = f"ft_{dataset}"
    elif model_type == 'fine-tuned':
        exp_name = f"ft_mozilla_{dataset}"


    exp_config = get_experiment_config()
    oversampling = exp_config['best_model_config'][run][model_name]['oversampling']
    embedding = exp_config['best_model_config'][run][model_name]['embedding']
    mode = exp_config['best_model_config'][run][model_name]['preprocess']
    config = exp_config['best_model_config'][run][model_name]['config']


    fine_tuned_model_path = f"./models/plm/0/roberta-base__BS_32__LR_3e-05__E_10__Oversample_nos.joblib"
    data_path = f"generalizability/dataset/{dataset}_with_embeddings.csv"
    model_to_save_path = f"generalizability/models/ml/{model_name}/{dataset}/{exp_name}"
    prediction_path = f"generalizability/predictions/ml/{model_name}/{dataset}/{exp_name}_predictions.csv"
    result_path = f"generalizability/results/ml/{model_name}/{dataset}/{exp_name}_results.csv"

    data = pd.read_csv(data_path)

    y_vals = []
    y_preds = []

    result_df = pd.DataFrame(columns=["preprocess","embedding","oversampling","model","run","accuracy","precision","recall","f1","TP","FP","TN","FN","param"])
    prediction_df = pd.DataFrame()

    folds = prepare_dataset(data)
    for i, fold in enumerate(folds):
        print(f"Fold {i + 1}/{len(folds)}")

        model_to_save_path_new = f"{model_to_save_path}/{i}/{model_name}"

        train_issue_id = fold['train']
        test_issue_id = fold['test']

        train_df = data[data['issue_id'].isin(train_issue_id)]
        test_df = data[data['issue_id'].isin(test_issue_id)]

        train_df_os = balance_classes(train_df, 'label', output_type='df')

        X_train = get_lm_embedding(train_df, embedding) # TODO Match Embedding for few rows
        y_train = train_df['label']
        X_test = get_lm_embedding(test_df, embedding)
        y_test = test_df['label']

        y_pred = ml_model(X_train , y_train, X_test, y_test, model_to_save_path_new, fine_tuned_model_path, run, mode, embedding, oversampling, model_name, config)

        y_vals.extend(y_test)
        y_preds.extend(y_pred)

        test_df = test_df.copy()
        test_df[model_name] = y_pred

        prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
    result = [mode,embedding,oversampling,model_name,run,accuracy,precision,recall,f1,TP,FP,TN,FN,config]
    result_df = pd.concat([result_df, pd.DataFrame([result], columns=result_df.columns)], ignore_index=True)

    print("=" * 50)
    print(f"Preprocessing Mode: {mode}")
    print(f"Embedding: {embedding}")
    print(f"Oversampling: {oversampling}")
    print(f"Model name: {model_name}")
    print(f"Oversampling: {oversampling}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print(f"Parameters: {config}")
    print("=" * 50)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result_df.to_csv(result_path, index=False)

    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    prediction_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved to: {prediction_path}")
