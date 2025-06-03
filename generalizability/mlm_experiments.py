import ast
from sources.utils import *
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
import joblib


def get_lm_embedding(data, embedder):
    train_embeddings = data[embedder + '_embedding'].apply(ast.literal_eval).tolist()
    train_embeddings = np.array(train_embeddings)
    return train_embeddings


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
    # Best ML model
    model_name = "SVC"
    # Best configuration for SVC of Mozilla experiments
    oversampling = False
    embedding = "gpt"
    mode = "none"
    config = {"C": 1, "class_weight": "balanced", "kernel": "rbf"}

    # We experiment with the datasets of two new projects
    projects = ['gnucash', 'chromium']
    # On these two datasets, we evaluate the performance of three trained models:
        # 1. mozilla: Model trained on the Mozilla data
        # 2. project: Model trained on the project (i.e., GunCash or Chromium) data
        # 3. mozilla-project: Model trained on both Mozilla and the project (i.e., GunCash or Chromium) data
    exp_types = ['mozilla', 'project', 'mozilla-project']

    # Read Mozilla data
    mozilla_data_path = 'dataset/embedding_data/actual_comment_data_with_embeddings.csv'
    mozilla_data_df = pd.read_csv(mozilla_data_path)

    X_train_mozilla = get_lm_embedding(mozilla_data_df, embedding)
    Y_train_mozilla = mozilla_data_df['label']
    print(f"# of Mozilla data samples: {len(X_train_mozilla)}")

    for project in projects:
        for exp_type in exp_types:
            if exp_type == 'mozilla':
                exp_name = "ft_mozilla"
            elif exp_type == 'project':
                exp_name = f"ft_{project}"
            elif exp_type == 'mozilla-project':
                exp_name = f"ft_mozilla_{project}"

            project_data_path = f"generalizability/dataset/{project}_with_embeddings.csv"
            model_to_save_path = f"generalizability/models/ml/{model_name}/{project}/{exp_name}"
            prediction_path = f"generalizability/predictions/ml/{model_name}/{project}/{exp_name}_predictions.csv"
            result_path = f"generalizability/results/ml/{model_name}/{project}/{exp_name}_results.csv"

            # Read project (i.e., GunCash or Chromium) data
            project_data = pd.read_csv(project_data_path)
            X_test_project = get_lm_embedding(project_data, embedding)
            y_test_project = project_data['label']

            y_vals = []
            y_preds = []

            result_df = pd.DataFrame(
                columns=["preprocess", "embedding", "oversampling", "model", "accuracy", "precision",
                         "recall", "f1", "TP", "FP", "TN", "FN", "param"])
            prediction_df = pd.DataFrame()

            # Run experiment 1
            if exp_type == 'mozilla':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                model_save_name = f"{model_name}_{project}_{exp_name}.joblib"
                model_to_save_path = os.path.join(model_to_save_path, model_save_name)
                y_pred = train_model_and_get_predictions(X_train_mozilla, Y_train_mozilla, X_test_project, model_name, config, model_to_save_path)
                y_preds.extend(y_pred)
                y_vals.extend(y_test_project)

            # Run experiment 2 or 3
            elif exp_type == 'project' or exp_type == 'mozilla-project':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                folds = prepare_dataset(project_data)
                for i, fold in enumerate(folds):
                    print(f"Fold {i + 1}/{len(folds)}")

                    model_save_name = f"{model_name}_{project}_{exp_name}_{i+1}.joblib"
                    model_to_save_path_new = os.path.join(model_to_save_path, model_save_name)

                    train_issue_id = fold['train']
                    test_issue_id = fold['test']
                    train_df = project_data[project_data['issue_id'].isin(train_issue_id)]
                    test_df = project_data[project_data['issue_id'].isin(test_issue_id)]

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
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print("=" * 50)

            # Save results
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_df.to_csv(result_path, index=False)
            # Save predictions
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            prediction_df.to_csv(prediction_path, index=False)
            print(f"Predictions saved to: {prediction_path}")

