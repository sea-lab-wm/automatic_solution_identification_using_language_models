import ast
import json
import os
import random
from multiprocessing import Pool

from joblib import Parallel, delayed, parallel_config
from utils import append_row_to_csv, calculate_results, get_experiment_config, prepare_data, get_train_test_data
from utils import dev_data_path, test_data_path, train_test_comment_data, dataset_fold_path, ml_model_base_path, embedding_comment_data, ml_prediction_comment_dir_exp2, train_data_path, val_data_path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import pandas as pd
import pandas as pd
import joblib
from b_preprocess_data import text_preprocess
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

# fixed-configuration rerun: config file, experiment name and where the prepared ML data is stored
ml_config_file = 'experiments_table4.json'  # all 48 Table 4 configurations; experiments_rq1_min.json has the 18 RQ1 ones
ml_exp = "rerun-ml-table4"
ml_data_path = f"/scratch/msun12/{ml_exp}/data"

# comment text lost in the October 2026 data update (now empty, so dropna removed the comment); restored to the
# text the 2024 results were computed with: dataset_construction/comment_data/actual_comment_data.csv at commit 4418c6e
restored_text_2024 = {(1769748, 6): "(In reply to Atila Butkovits from comment #4)"}

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


def parse_embedding(value):
    # same parser as get_lm_embedding, so the embedding values are exactly those of the original pipeline
    return ast.literal_eval(value)


def prepare_data_for_ml(model_configs, data_dir, runs=range(10), n_jobs=16):
    # Prepare the ML data once for all folds instead of per fold: every comment's preprocessed text and
    # LM embedding is stored a single time, plus the dev/test rows of each fold. Folds are built from the
    # embedding data and the split file exactly like utils.prepare_data, without writing the fold CSVs.
    data_configs = sorted({(c['preprocess'], c['embedding'], c['oversampling'])
                           for run in runs for c in model_configs[str(run)]})
    modes = sorted({mode for mode, embedding, _ in data_configs if embedding not in ["bert", "llama", "gpt"]})
    lm_embeddings = sorted({embedding for _, embedding, _ in data_configs if embedding in ["bert", "llama", "gpt"]})
    print(f"Preparing data for {data_configs} in {data_dir}")

    data = pd.read_csv(embedding_comment_data)
    for (issue_id, text_id), text in restored_text_2024.items():
        row = (data['issue_id'] == issue_id) & (data['text_id'] == text_id)
        if row.sum() == 1 and data.loc[row, 'text'].isna().all():
            data.loc[row, 'text'] = text
            print(f"Restored 2024 text of comment ({issue_id}, {text_id}): {text!r}")
    with open(train_test_comment_data, 'r') as f:
        indexes = json.load(f)

    fold_keys = {}
    for run in runs:
        test_df, dev_df = get_train_test_data(data, indexes, run)
        # same as utils.process_run: rows with any missing value are dropped
        test_df, dev_df = test_df.dropna(), dev_df.dropna()
        fold_keys[run] = {'dev': list(zip(dev_df['issue_id'], dev_df['text_id'])),
                          'test': list(zip(test_df['issue_id'], test_df['text_id']))}
        print(f"Run {run}: {len(dev_df)} dev rows {dev_df['label'].value_counts().to_dict()}, "
              f"{len(test_df)} test rows {test_df['label'].value_counts().to_dict()}")

    used = {key for fold in fold_keys.values() for split in fold.values() for key in split}
    comments = data[[key in used for key in zip(data['issue_id'], data['text_id'])]].reset_index(drop=True)
    position = {key: i for i, key in enumerate(zip(comments['issue_id'], comments['text_id']))}
    if len(position) != len(comments):
        raise ValueError("duplicate (issue_id, text_id) keys in the comment data")
    folds = {run: {split: np.array([position[key] for key in keys]) for split, keys in fold.items()}
             for run, fold in fold_keys.items()}

    os.makedirs(data_dir, exist_ok=True)
    with Pool(n_jobs) as pool:
        for mode in modes:
            comments[f'text_{mode}'] = pool.starmap(text_preprocess, [(text, mode) for text in comments['text']],
                                                    chunksize=64)
            print(f"Preprocessed text with mode '{mode}'")
        for embedding in lm_embeddings:
            matrix = np.array(pool.map(parse_embedding, comments[f'{embedding}_embedding'], chunksize=64))
            np.save(os.path.join(data_dir, f'embedding_{embedding}.npy'), matrix)
            print(f"Saved {embedding} embeddings {matrix.shape} to {data_dir}")

    columns = ['issue_id', 'text_id', 'text', 'code', 'label'] + [f'text_{mode}' for mode in modes]
    full_path = os.path.join(data_dir, 'ml_data.joblib')
    joblib.dump({'comments': comments[columns], 'folds': folds, 'data_configs': data_configs}, full_path)
    print(f"Saved {len(comments)} comments and {len(folds)} folds to {full_path}")


def load_ml_data(data_dir):
    # small file (text, labels, fold indices); a plain function so joblib workers can receive it
    # when this file is run as a script (an lru_cache wrapper cannot be pickled from __main__)
    return joblib.load(os.path.join(data_dir, 'ml_data.joblib'))


def get_test_frame(data_dir, run):
    ml_data = load_ml_data(data_dir)
    test_index = ml_data['folds'][run]['test']
    return ml_data['comments'].iloc[test_index][['issue_id', 'text_id', 'text', 'code', 'label']].reset_index(drop=True)


def get_fold_data(data_dir, run, mode, embedding, oversampling):
    ml_data = load_ml_data(data_dir)
    comments = ml_data['comments']
    dev_index, test_index = ml_data['folds'][run]['dev'], ml_data['folds'][run]['test']
    y_dev = comments['label'].iloc[dev_index].reset_index(drop=True)
    y_test = comments['label'].iloc[test_index].reset_index(drop=True)

    if embedding in ["bert", "llama", "gpt"]:
        matrix = np.load(os.path.join(data_dir, f'embedding_{embedding}.npy'), mmap_mode='r')
        X_dev_embedding, X_test_embedding = np.asarray(matrix[dev_index]), np.asarray(matrix[test_index])
    else:
        X_test_embedding, X_dev_embedding = get_embedding(comments.iloc[test_index], comments.iloc[dev_index], mode)

    if oversampling:
        X_dev_embedding, y_dev = oversampling_data(X_dev_embedding, y_dev)

    return {
        'X_dev_embedding': X_dev_embedding,
        'X_test_embedding': X_test_embedding,
        'y_dev': y_dev,
        'y_test': y_test,
    }

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
        return SVC(random_state=42, probability=True)
    elif clf_name == "Random_Forest":
        return RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown classifier name: {clf_name}")


def get_positive_class_confidence(model, X_test):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support probability prediction")

    probabilities = np.asarray(model.predict_proba(X_test))
    positive_class_index = list(model.classes_).index(1)

    return probabilities[:, positive_class_index]


def get_tuned_model_prediction(run, X_test, X_train, clf, model, config, y_train, model_save_path, model_dir=ml_model_base_path):
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
    y_confidence = get_positive_class_confidence(best_model, X_test)
    
    save_model(best_model, os.path.join(model_dir, f"run_{run}_{model_save_path}"))
    return config, y_pred, y_confidence

def ml_model(data_dict, run, mode, embedding, oversampling, model, config, model_dir):
    X_train_embedding = data_dict['X_dev_embedding']
    X_test_embedding = data_dict['X_test_embedding']
    y_train = data_dict['y_dev']
    y_test = data_dict['y_test']

    set_ml_seed(42)
    
    clf = getBaseModel(model["model_name"])
    model_save_path = f'model_{model["model_name"]}_preprocess_{mode}__embedding_{embedding}__oversampling_{"os" if oversampling else "nos"}.joblib'
    param, y_pred, y_confidence = get_tuned_model_prediction(
        run, 
        X_test_embedding,
        X_train_embedding,
        clf,
        model,
        config,
        y_train,
        model_save_path,
        model_dir,
    )

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)

    result = [mode, embedding, oversampling, model['model_name'], run, accuracy, precision, recall, f1,
              TP, FP, TN, FN, param]

    print("=" * 50)
    print(f"Preprocessing Mode: {mode}")
    print(f"Embedding: {embedding}")
    print(f"Oversampling: {oversampling}")
    print(f"Model name: {model['model_name']}")
    print(f"Run: {run + 1}")
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

    # results are written by the parent process (ml_run), so parallel workers never write the same file
    return result, y_pred, y_confidence

def process_config(mode, embedding, oversampling, model, config, path, run, model_dir, data_dir, reference=None):
    if embedding in ["bert", "llama", "gpt"] and mode in ["essential", "all"]:
        return None

    column_name = '_'.join(
        [mode, embedding, "os" if oversampling else "nos", model['model_name'],
         str(run)])
    try:
        print(f"Preprocess: {mode}, Embedding: {embedding}, Oversampling: {oversampling}, Model: {model['model_name']}, Run: {run}")
        data_dict = get_fold_data(data_dir, run, mode, embedding, oversampling)
        result, y_pred, y_confidence = ml_model(data_dict, run, mode, embedding, oversampling, model, config, model_dir)

    except Exception as e:
        print(f"An error occurred in {column_name}: {e}")
        return {'run': run, 'column_name': column_name, 'error': f"{type(e).__name__}: {e}"}
    return {'run': run, 'column_name': column_name, 'result': result, 'y_pred': y_pred,
            'y_confidence': y_confidence, 'reference': reference}


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


def ml_run(exp_config, run_best_config=False, n_jobs=-1, threads_per_worker=1, exp=ml_exp, data_dir=ml_data_path):
    # n_jobs=-1 uses one worker per CPU core; keep n_jobs * threads_per_worker <= number of cores
    # separate output folders for this experiment: results, predictions and models
    result_dir = f"results/{exp}/ml"
    prediction_dir = f"results/{exp}/predictions/ml"
    model_dir = f"/scratch/msun12/model/{exp}"
    result_path = os.path.join(result_dir, "comment_data_results.csv")
    comparison_path = os.path.join(result_dir, "reference_comparison.csv")
    prediction_path_template = os.path.join(prediction_dir, "run_{run}.csv")
    confidence_prediction_path_template = os.path.join(prediction_dir, "run_{run}_with_confidence.csv")
    for directory in [result_dir, prediction_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
    print(f"Result Path: {result_path}\nPrediction Dir: {prediction_dir}\nModel Dir: {model_dir}\nData Dir: {data_dir}")

    result_header = ["preprocess", "embedding", "oversampling", "model", "run", "accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN", "param"]
    if not os.path.exists(result_path):
        append_row_to_csv(result_path, result_header)
        print(f"{result_header} is saved to {result_path}")
    if not os.path.exists(comparison_path):
        append_row_to_csv(comparison_path, ["run", "config", "TP", "FP", "TN", "FN", "f1",
                                            "TP_ref", "FP_ref", "TN_ref", "FN_ref", "f1_ref", "match"])

    # RQ1 minimum set: 6 models x {tfidf (preprocess=all), llama, gpt} x no data balancing
    selected_models = [
        "Logistic_Regression",
        "Naive_Bayes",
        "K_Nearest_Neighbors",
        "Decision_Tree",
        "SVC",
        "Random_Forest"
    ]
    data_balance = [
        "os",
        "nos"
    ]
    embeddings = [
        "tfidf",
        "bert",
        "llama",
        "gpt"
    ]
    runs = range(10)

    def is_selected(model_name, embedding, oversampling):
        return (model_name in selected_models and embedding in embeddings
                and ("os" if oversampling else "nos") in data_balance)

    # collect the tasks of every run first, so runs and the tasks within them share one worker pool
    tasks = []
    test_predictions = {}
    prepared_runs = load_ml_data(data_dir)['folds']
    for run in runs:
        if run not in prepared_runs:
            print(f"Skipping run {run}: not prepared in {data_dir} (prepared runs: {sorted(prepared_runs)})")
            continue
        path = f"{dataset_fold_path}/{run}/comment_test_data.csv"
        test_predictions[run] = get_test_frame(data_dir, run)

        if run_best_config and 'model_configs' in exp_config:
            # list of fixed configurations per fold, e.g. the 48 Table 4 configurations in experiments_table4.json
            run_tasks = []
            for model_details in exp_config['model_configs'][str(run)]:
                model = {'model_name': model_details['model_name']}
                preprocess = model_details['preprocess']
                embedding = model_details['embedding']
                oversampling = model_details['oversampling']
                config = model_details['config']
                reference = model_details.get('reference')
                run_tasks.append((preprocess, embedding, oversampling, model, config, path, run, model_dir, data_dir, reference))
        elif run_best_config:
            run_tasks = []
            for model in exp_config['models']:
                model_details = exp_config['best_model_config'][str(run)][model['model_name']]
                preprocess = model_details['preprocess']
                embedding = model_details['embedding']
                oversampling = model_details['oversampling']
                config = model_details['config']
                run_tasks.append((preprocess, embedding, oversampling, model, config, path, run, model_dir, data_dir, None))
        else:
            run_tasks = [
                (mode, embedding, oversampling, model, None, path, run, model_dir, data_dir, None)
                for model in exp_config['models']
                for oversampling in exp_config['oversampling']
                for embedding in exp_config['embedding']
                for mode in exp_config['preprocess']
                if not (embedding in ["bert", "llama", "gpt"] and mode in ["essential", "all"])
            ]

        tasks.extend(task for task in run_tasks if is_selected(task[3]['model_name'], task[1], task[2]))

    workers = os.cpu_count() if n_jobs == -1 else n_jobs
    print(f"Running {len(tasks)} tasks for runs {list(runs)} with {workers} parallel workers, "
          f"{threads_per_worker} BLAS thread(s) each")

    failed = []
    matched = 0
    # inner_max_num_threads caps BLAS/OpenMP threads per worker so parallel workers do not oversubscribe the CPUs
    with parallel_config(backend="loky", inner_max_num_threads=threads_per_worker):
        outputs = Parallel(n_jobs=n_jobs, return_as="generator_unordered", verbose=10)(
            delayed(process_config)(*task) for task in tasks
        )
        for output in outputs:
            if output is None:
                continue
            run, column_name = output['run'], output['column_name']
            if 'error' in output:
                failed.append((run, column_name, output['error']))
                continue

            result = output['result']
            append_row_to_csv(result_path, result)
            print(f"Results are saved to {result_path}")

            test_predictions[run][column_name] = output['y_pred']
            test_predictions[run][column_name + "_confidence"] = output['y_confidence']
            standard_prediction_path = prediction_path_template.format(run=run)
            confidence_prediction_path = confidence_prediction_path_template.format(run=run)
            test_predictions[run].to_csv(standard_prediction_path, index=False)
            test_predictions[run].to_csv(confidence_prediction_path, index=False)
            print(f"Test Prediction Saved to {standard_prediction_path}")
            print(f"Confidence Prediction Saved to {confidence_prediction_path}")

            reference = output['reference']
            if reference:
                TP, FP, TN, FN = result[9:13]
                match = (TP, FP, TN, FN) == (reference['TP'], reference['FP'], reference['TN'], reference['FN'])
                matched += match
                append_row_to_csv(comparison_path, [run, column_name, TP, FP, TN, FN, round(result[8] / 100, 4),
                                                    reference['TP'], reference['FP'], reference['TN'], reference['FN'],
                                                    reference['f1'], match])
                print(f"{column_name}: {'MATCH' if match else 'MISMATCH'} with reference "
                      f"(TP {TP} FP {FP} TN {TN} FN {FN} vs TP {reference['TP']} FP {reference['FP']} "
                      f"TN {reference['TN']} FN {reference['FN']})")

    print(f"Finished: {len(tasks) - len(failed)}/{len(tasks)} tasks succeeded, "
          f"{matched} matched the reference confusion matrix")
    for run, column_name, error in failed:
        print(f"FAILED run {run} {column_name}: {error}")


if __name__ == "__main__":
    exp_config = get_experiment_config()

    # fixed configurations x 10 folds with the tuned hyperparameters from the original run (see ml_config_file)
    with open(ml_config_file, 'r') as file:
        exp_config['model_configs'] = json.load(file)['model_configs']

    runs = range(10)

    # prepare_data(runs, embedding_comment_data, train_test_comment_data)

    prepare_data_for_ml(exp_config['model_configs'], ml_data_path, runs)  # writes the shared ML data to ml_data_path

    ml_run(exp_config, True)  # Set to True to run with best model configurations, False for all configurations
