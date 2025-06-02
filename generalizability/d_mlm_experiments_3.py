import ast
import os

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

                    X_dev_embedding, X_test_embedding, y_dev, y_test = get_splitted_data(dev_df, test_df, mode,
                                                                                         embedding, oversampling)
                    X_train_embedding, X_val_embedding, y_train, y_val = get_splitted_data(train_df, val_df, mode,
                                                                                           embedding, oversampling)

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

                    print(
                        f"Data saved to {full_path} for run {run}, mode {mode}, embedding {embedding}, oversampling {oversampling}")


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


def train_model_and_get_predictions(X_train, y_train, X_test, model_name, config, model_to_save_path):
    clf = getBaseModel(model_name)
    model = clf.set_params(**config)

    # Training
    model.fit(X_train, y_train)

    # Testing
    y_pred = model.predict(X_test)

    save_model(model, model_to_save_path)
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


def get_train_test_df(run, data_path, index_path):
    data = clean_dataframe(data_path)

    print(data['label'].value_counts())

    with open(index_path, 'r') as f:
        indexes = json.load(f)

    test_df, train_df = get_train_test_data(data, indexes, run)

    test_df = test_df.dropna()
    train_df = train_df.dropna()

    return train_df, test_df



if __name__ == "__main__":
    # Best ML model
    model_name = "SVC"
    # Best configuration for SVC of Mozilla experiments
    oversampling = False
    embedding = "gpt"
    mode = "none"
    config = {"C": 1, "class_weight": "balanced", "kernel": "rbf"}

    # datasets = ['chromium', 'gnucash']
    # model_types = ['base', 'fine-tuned']
    projects = ['gnucash']
    exp_types = ['mozilla', 'project', 'mozilla-project']

    mozilla_data_path = 'dataset/embedding_data/actual_comment_data_with_embeddings.csv'
    # mozilla_data_split_path = 'dataset/solution_identification_data/train_test_split.json'
    #
    # mozilla_train_df, mozilla_test_df = get_train_test_df(run, mozilla_data_path, mozilla_data_split_path)
    #
    # X_train_mozilla = get_lm_embedding(mozilla_train_df, embedding)
    # Y_train_mozilla = mozilla_train_df['label']

    mozilla_data_df = pd.read_csv(mozilla_data_path)

    X_train_mozilla = get_lm_embedding(mozilla_data_df, embedding)
    Y_train_mozilla = mozilla_data_df['label']

    print(f"# of training samples: {len(X_train_mozilla)}")

    for project in projects:
        for exp_type in exp_types:

            if exp_type == 'mozilla':
                exp_name = "ft_mozilla"
            elif exp_type == 'project':
                exp_name = f"ft_{project}"
            elif exp_type == 'mozilla-project':
                exp_name = f"ft_mozilla_{project}"

            project_data_path = f"generalizability/dataset/{project}_with_embeddings.csv"
            model_to_save_path = f"generalizability/models/ml/{model_name}/{exp_name}"
            prediction_path = f"generalizability/predictions/ml/{model_name}/{project}/{exp_name}_predictions.csv"
            result_path = f"generalizability/results/ml/{model_name}/{project}/{exp_name}_results.csv"

            project_data = pd.read_csv(project_data_path)

            y_vals = []
            y_preds = []

            result_df = pd.DataFrame(
                columns=["preprocess", "embedding", "oversampling", "model", "accuracy", "precision",
                         "recall", "f1", "TP", "FP", "TN", "FN", "param"])
            prediction_df = pd.DataFrame()

            X_test_project = get_lm_embedding(project_data, embedding)
            y_test_project = project_data['label']

            if exp_type == 'mozilla':
                y_preds = train_model_and_get_predictions(X_train_mozilla, Y_train_mozilla, X_test_project, model_name, config, model_to_save_path)

            elif exp_type == 'project' or exp_type == 'mozilla-project':
                folds = prepare_dataset(project_data)
                for i, fold in enumerate(folds):
                    print(f"Fold {i + 1}/{len(folds)}")

                    model_to_save_path_new = f"{model_to_save_path}/{i}"

                    train_issue_id = fold['train']
                    test_issue_id = fold['test']

                    train_df = project_data[project_data['issue_id'].isin(train_issue_id)]
                    test_df = project_data[project_data['issue_id'].isin(test_issue_id)]

                    # train_df_os = balance_classes(train_df, 'label', output_type='df')

                    X_train = get_lm_embedding(train_df, embedding)  # TODO Match Embedding for few rows
                    Y_train = train_df['label']

                    if exp_type == 'mozilla-project':
                        X_train = np.concatenate((X_train_mozilla, X_train))
                        Y_train = pd.concat([Y_train_mozilla, Y_train])

                    X_test = get_lm_embedding(test_df, embedding)
                    y_test = test_df['label']

                    # Training and evaluation
                    y_pred = train_model_and_get_predictions(X_train, Y_train, X_test, model_name, config, model_to_save_path_new)

                    y_vals.extend(y_test)
                    y_preds.extend(y_pred)

                    test_df = test_df.copy()
                    test_df[model_name] = y_pred

                    prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

            FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
            result = [mode, embedding, oversampling, model_name, accuracy, precision, recall, f1, TP, FP, TN,
                      FN, config]
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

