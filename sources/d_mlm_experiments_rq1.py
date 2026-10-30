"""Re-run the RQ1 traditional ML configurations on all 10 folds.

Follows sources/d_mlm_experiments.py (same split, features, base models and
metrics) but, instead of grid search, uses the tuned hyperparameters recorded
in the reference results CSV for every (configuration, fold). Results,
per-comment predictions/confidences, fitted models, timings and a comparison
against the reference metrics are written to a new output folder.

Default scope (RQ1 claims): 6 models x {TF-IDF with preprocessing, gpt, llama},
no data balancing, folds 0-9 -> 18 configurations x 10 folds.

Usage (from the repository root):
    python sources/d_mlm_experiments_rq1.py --runs 0 --smoke --out_dir <dir>
    python sources/d_mlm_experiments_rq1.py --runs 0 --configs none_gpt_SVC
    python sources/d_mlm_experiments_rq1.py --workers 16
"""
import argparse
import ast
import datetime
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from b_preprocess_data import text_preprocess
from utils import calculate_results, get_train_test_data, embedding_comment_data, train_test_comment_data

reference_result_path = "results/ml/comment_data_results copy.csv"
default_out_dir = "results/rq1_ml_rerun"
default_model_dir = "/scratch/msun12/model/rq1_ml_rerun"

# (preprocess, embedding) pairs: TF-IDF with full preprocessing, Gpt_emb, Llama_emb
default_representations = [("all", "tfidf"), ("none", "gpt"), ("none", "llama")]
default_models = ["Logistic_Regression", "K_Nearest_Neighbors", "Naive_Bayes",
                  "SVC", "Decision_Tree", "Random_Forest"]
lm_embeddings = ["bert", "llama", "gpt"]

result_header = ["preprocess", "embedding", "oversampling", "model", "run", "accuracy", "precision", "recall",
                 "f1", "TP", "FP", "TN", "FN", "param"]

# Populated in the parent process before the worker pool forks
DATA = None
FOLD_KEYS = None
EMBEDDINGS = {}
ARGS = None


def log(message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def getBaseModel(clf_name):
    # identical to sources/d_mlm_experiments.py
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
    probabilities = np.asarray(model.predict_proba(X_test))
    positive_class_index = list(model.classes_).index(1)
    return probabilities[:, positive_class_index]


def config_name(mode, embedding, oversampling, model_name):
    return '_'.join([mode, embedding, "os" if oversampling else "nos", model_name])


def model_file_name(model_name, mode, embedding, oversampling):
    return f'model_{model_name}_preprocess_{mode}__embedding_{embedding}__oversampling_{"os" if oversampling else "nos"}.joblib'


def git_info():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run(["git", "rev-parse", "HEAD"]), "status": run(["git", "status", "--short"])}


def load_reference(path):
    ref = pd.read_csv(path)
    ref["param_dict"] = ref["param"].apply(ast.literal_eval)
    return ref


def load_data(runs):
    """Rebuild each fold exactly like utils.prepare_data / process_run (merge on split ids + dropna)."""
    global DATA, FOLD_KEYS, EMBEDDINGS

    log(f"Loading {embedding_comment_data}")
    data = pd.read_csv(embedding_comment_data)
    log(f"Loaded {len(data)} rows, columns: {list(data.columns)}")
    log(f"Label distribution: {data['label'].value_counts().to_dict()}")

    with open(train_test_comment_data, 'r') as f:
        indexes = json.load(f)

    FOLD_KEYS = {}
    used = set()
    for run in runs:
        test_df, dev_df = get_train_test_data(data, indexes, run)
        n_test, n_dev = len(test_df), len(dev_df)
        test_df = test_df.dropna()
        dev_df = dev_df.dropna()
        log(f"Run {run}: dev {n_dev}->{len(dev_df)} rows, test {n_test}->{len(test_df)} rows after dropna; "
            f"dev labels {dev_df['label'].value_counts().to_dict()}, test labels {test_df['label'].value_counts().to_dict()}")
        FOLD_KEYS[run] = {
            "dev": list(zip(dev_df['issue_id'], dev_df['text_id'])),
            "test": list(zip(test_df['issue_id'], test_df['text_id'])),
        }
        used.update(FOLD_KEYS[run]["dev"])
        used.update(FOLD_KEYS[run]["test"])

    data = data[[k in used for k in zip(data['issue_id'], data['text_id'])]].copy()
    data = data.set_index(['issue_id', 'text_id'], drop=False)
    assert data.index.is_unique, "duplicate (issue_id, text_id) keys in comment data"
    DATA = data
    log(f"{len(DATA)} distinct comments used across runs {list(runs)}")


def prepare_features(representations, workers):
    global DATA, EMBEDDINGS

    modes = sorted({mode for mode, embedding in representations if embedding not in lm_embeddings})
    for mode in modes:
        if mode == 'none':
            DATA['text_none'] = DATA['text']
            continue
        log(f"Preprocessing text with mode '{mode}' ({len(DATA)} comments, {workers} workers)")
        start = time.time()
        with Pool(workers) as pool:
            DATA[f'text_{mode}'] = pool.starmap(text_preprocess, [(t, mode) for t in DATA['text']], chunksize=64)
        log(f"Preprocessing done in {time.time() - start:.1f}s")

    for embedding in sorted({e for _, e in representations if e in lm_embeddings}):
        log(f"Parsing {embedding} embeddings")
        start = time.time()
        with Pool(workers) as pool:
            parsed = pool.map(ast.literal_eval, DATA[embedding + '_embedding'], chunksize=64)
        EMBEDDINGS[embedding] = pd.Series(list(np.array(parsed)), index=DATA.index)
        log(f"{embedding}: dim {len(parsed[0])}, parsed in {time.time() - start:.1f}s")

    drop_cols = [c for c in DATA.columns if c.endswith('_embedding')]
    DATA.drop(columns=drop_cols, inplace=True)


def subsample(keys, labels, n, seed=42):
    rng = np.random.RandomState(seed)
    keys = np.array(keys, dtype=object)
    labels = np.asarray(labels)
    picked = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        take = max(2, int(round(n * len(idx) / len(labels))))
        picked.extend(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return [tuple(k) for k in keys[np.sort(picked)]]


def get_fold_data(run, mode, embedding, smoke):
    dev_keys = FOLD_KEYS[run]["dev"]
    test_keys = FOLD_KEYS[run]["test"]
    if smoke:
        dev_keys = subsample(dev_keys, DATA.loc[dev_keys, 'label'].values, smoke)
        test_keys = subsample(test_keys, DATA.loc[test_keys, 'label'].values, max(smoke // 4, 20))

    dev_df = DATA.loc[dev_keys]
    test_df = DATA.loc[test_keys]
    y_train, y_test = dev_df['label'], test_df['label']

    vectorizer = None
    if embedding in lm_embeddings:
        X_train = np.array(EMBEDDINGS[embedding].loc[dev_keys].tolist())
        X_test = np.array(EMBEDDINGS[embedding].loc[test_keys].tolist())
    else:
        # same as get_embedding() in d_mlm_experiments.py
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(dev_df[f'text_{mode}']).toarray()
        X_test = vectorizer.transform(test_df[f'text_{mode}']).toarray()

    return X_train, X_test, y_train, y_test, test_df, vectorizer


def free_gb(path):
    return shutil.disk_usage(path).free / 1024 ** 3


def run_task(task):
    run, mode, embedding, oversampling, model_name, config = task
    name = config_name(mode, embedding, oversampling, model_name)
    info = {"run": run, "config": name, "pid": os.getpid(), "start": datetime.datetime.now().isoformat()}
    try:
        with threadpool_limits(limits=ARGS.threads):
            t0 = time.time()
            X_train, X_test, y_train, y_test, test_df, vectorizer = get_fold_data(run, mode, embedding, ARGS.smoke)
            info["feature_seconds"] = round(time.time() - t0, 3)
            info.update({"n_train": len(y_train), "n_test": len(y_test), "n_features": X_train.shape[1],
                         "train_pos": int((y_train == 1).sum()), "test_pos": int((y_test == 1).sum())})

            np.random.seed(42)
            clf = getBaseModel(model_name).set_params(**config)
            t0 = time.time()
            clf.fit(X_train, y_train)
            info["fit_seconds"] = round(time.time() - t0, 3)

            t0 = time.time()
            y_pred = clf.predict(X_test)
            y_confidence = get_positive_class_confidence(clf, X_test)
            info["predict_seconds"] = round(time.time() - t0, 3)

        FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)
        result = [mode, embedding, oversampling, model_name, run, accuracy, precision, recall, f1,
                  TP, FP, TN, FN, config]

        info["model_bytes"] = 0
        if ARGS.save_models:
            model_dir = os.path.join(ARGS.model_dir, f"run_{run}")
            os.makedirs(model_dir, exist_ok=True)
            if free_gb(model_dir) < ARGS.min_free_gb:
                info["model_saved"] = f"skipped: free disk < {ARGS.min_free_gb} GB"
            else:
                t0 = time.time()
                model_path = os.path.join(model_dir, model_file_name(model_name, mode, embedding, oversampling))
                joblib.dump(clf, model_path, compress=3)
                info["model_bytes"] = os.path.getsize(model_path)
                info["save_seconds"] = round(time.time() - t0, 3)
                info["model_saved"] = model_path
                if vectorizer is not None:
                    vec_path = os.path.join(model_dir, f"tfidf_vectorizer_preprocess_{mode}.joblib")
                    if not os.path.exists(vec_path):
                        joblib.dump(vectorizer, vec_path, compress=3)

        predictions = pd.DataFrame({
            "issue_id": test_df['issue_id'].values,
            "text_id": test_df['text_id'].values,
            f"{name}_{run}": y_pred,
            f"{name}_{run}_confidence": y_confidence,
        })
        info["end"] = datetime.datetime.now().isoformat()
        return {"ok": True, "result": result, "info": info, "predictions": predictions}
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
        return {"ok": False, "info": info}


def build_tasks(reference, runs, representations, models, only_configs):
    tasks = []
    for run in runs:
        for mode, embedding in representations:
            for model_name in models:
                name = config_name(mode, embedding, False, model_name)
                if only_configs and name not in only_configs and f"{mode}_{embedding}_{model_name}" not in only_configs:
                    continue
                row = reference[(reference.preprocess == mode) & (reference.embedding == embedding)
                                & (reference.oversampling == False) & (reference.model == model_name)
                                & (reference.run == run)]
                if len(row) != 1:
                    raise ValueError(f"Expected 1 reference row for {name} run {run}, found {len(row)}")
                tasks.append((run, mode, embedding, False, model_name, row.iloc[0]["param_dict"]))
    # slow model families first so the pool does not finish on a long tail
    order = {"SVC": 0, "Random_Forest": 1, "K_Nearest_Neighbors": 2, "Logistic_Regression": 3,
             "Decision_Tree": 4, "Naive_Bayes": 5}
    tasks.sort(key=lambda t: (order.get(t[4], 9), t[2] != "tfidf", t[0]))
    return tasks


def completed_tasks(details_path):
    if not os.path.exists(details_path):
        return set()
    details = pd.read_csv(details_path)
    details = details[details["status"] == "ok"]
    return set(zip(details["run"], details["config"]))


def compare_with_reference(reference, results_path, out_dir):
    if not os.path.exists(results_path):
        return
    ours = pd.read_csv(results_path)
    keys = ["preprocess", "embedding", "oversampling", "model", "run"]
    merged = ours.merge(reference.drop(columns=["param_dict"]), on=keys, how="left", suffixes=("", "_ref"))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        merged[metric] = merged[metric] / 100
        merged[f"{metric}_diff"] = (merged[metric] - merged[f"{metric}_ref"]).round(4)
    merged["confusion_match"] = ((merged["TP"] == merged["TP_ref"]) & (merged["FP"] == merged["FP_ref"])
                                 & (merged["TN"] == merged["TN_ref"]) & (merged["FN"] == merged["FN_ref"]))
    merged.to_csv(os.path.join(out_dir, "comparison_with_reference.csv"), index=False)

    summary = merged.groupby(["preprocess", "embedding", "model"]).agg(
        folds=("run", "count"),
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"), f1_std=("f1", "std"),
        precision_ref=("precision_ref", "mean"), recall_ref=("recall_ref", "mean"), f1_ref=("f1_ref", "mean"),
        folds_matching_confusion=("confusion_match", "sum"),
        max_abs_f1_diff=("f1_diff", lambda s: s.abs().max()),
    ).round(4).reset_index()
    summary.to_csv(os.path.join(out_dir, "summary_by_config.csv"), index=False)
    log(f"Reference comparison: {int(merged['confusion_match'].sum())}/{len(merged)} (config, fold) pairs "
        f"match the reference confusion matrix exactly")
    log("Per-configuration summary (fraction metrics, mean over folds):\n" + summary.to_string(index=False))


def assemble_predictions(out_dir, runs):
    per_config_dir = os.path.join(out_dir, "predictions", "per_config")
    for run in runs:
        run_dir = os.path.join(per_config_dir, f"run_{run}")
        if not os.path.isdir(run_dir):
            continue
        keys = FOLD_KEYS[run]["test"]
        base = DATA.loc[keys, ['issue_id', 'text_id', 'text', 'code', 'label']].reset_index(drop=True)
        with_confidence = base.copy()
        for file in sorted(os.listdir(run_dir)):
            pred = pd.read_csv(os.path.join(run_dir, file))
            with_confidence = with_confidence.merge(pred, on=['issue_id', 'text_id'], how='left')
        pred_cols = [c for c in with_confidence.columns if c not in base.columns and not c.endswith("_confidence")]
        with_confidence[base.columns.tolist() + pred_cols].to_csv(
            os.path.join(out_dir, "predictions", f"run_{run}.csv"), index=False)
        with_confidence.to_csv(os.path.join(out_dir, "predictions", f"run_{run}_with_confidence.csv"), index=False)
    log(f"Wide prediction files written to {os.path.join(out_dir, 'predictions')}")


def main():
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="restrict to configs, e.g. none_gpt_SVC or all_tfidf_nos_Random_Forest")
    parser.add_argument("--out_dir", default=default_out_dir)
    parser.add_argument("--model_dir", default=default_model_dir, help="where fitted models are saved")
    parser.add_argument("--reference", default=reference_result_path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2, help="BLAS/OpenMP threads per worker")
    parser.add_argument("--smoke", type=int, default=0,
                        help="if > 0, subsample this many training comments (stratified) for a quick code-path test")
    parser.add_argument("--no_save_models", dest="save_models", action="store_false")
    parser.add_argument("--min_free_gb", type=float, default=5.0,
                        help="skip saving a model when the disk has less free space than this")
    ARGS = parser.parse_args()

    out_dir = ARGS.out_dir
    os.makedirs(os.path.join(out_dir, "predictions", "per_config"), exist_ok=True)
    results_path = os.path.join(out_dir, "comment_data_results.csv")
    details_path = os.path.join(out_dir, "run_details.csv")
    errors_path = os.path.join(out_dir, "errors.jsonl")

    started = time.time()
    metadata = {
        "script": os.path.abspath(__file__),
        "argv": sys.argv,
        "args": vars(ARGS),
        "start": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "joblib": joblib.__version__},
        "git": git_info(),
        "inputs": {
            "embedding_data": embedding_comment_data,
            "split": train_test_comment_data,
            "reference": ARGS.reference,
        },
        "free_disk_gb_at_start": round(free_gb(out_dir), 2),
        "model_dir": ARGS.model_dir,
    }
    with open(os.path.join(out_dir, f"metadata_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Run metadata: {json.dumps(metadata, default=str)}")

    reference = load_reference(ARGS.reference)
    tasks = build_tasks(reference, ARGS.runs, default_representations, default_models, ARGS.configs)
    done = completed_tasks(details_path)
    pending = [t for t in tasks if (t[0], config_name(t[1], t[2], t[3], t[4])) not in done]
    log(f"{len(tasks)} tasks requested, {len(tasks) - len(pending)} already completed, {len(pending)} to run")

    load_data(ARGS.runs)
    prepare_features(default_representations, ARGS.workers)

    if not os.path.exists(results_path):
        pd.DataFrame(columns=result_header).to_csv(results_path, index=False)

    n_done = 0
    with Pool(min(ARGS.workers, max(len(pending), 1))) as pool:
        for output in pool.imap_unordered(run_task, pending):
            n_done += 1
            info = output["info"]
            if output["ok"]:
                pd.DataFrame([output["result"]], columns=result_header).to_csv(
                    results_path, mode="a", header=False, index=False)
                pred_dir = os.path.join(out_dir, "predictions", "per_config", f"run_{info['run']}")
                os.makedirs(pred_dir, exist_ok=True)
                output["predictions"].to_csv(os.path.join(pred_dir, f"{info['config']}.csv"), index=False)
                info["status"] = "ok"
                r = output["result"]
                info.update({"precision": r[6], "recall": r[7], "f1": r[8], "TP": r[9], "FP": r[10],
                             "TN": r[11], "FN": r[12], "param": str(r[13])})
                ref = reference[(reference.preprocess == r[0]) & (reference.embedding == r[1])
                                & (reference.oversampling == r[2]) & (reference.model == r[3])
                                & (reference.run == r[4])].iloc[0]
                info["f1_ref"] = ref["f1"]
                info["ref_match"] = bool((ref["TP"], ref["FP"], ref["TN"], ref["FN"]) == tuple(r[9:13]))
                log(f"[{n_done}/{len(pending)}] run {info['run']} {info['config']}: F1 {r[8]:.2f} "
                    f"(TP {r[9]} FP {r[10]} TN {r[11]} FN {r[12]}), fit {info['fit_seconds']}s, "
                    f"predict {info['predict_seconds']}s, model {info['model_bytes'] / 1024 ** 2:.1f} MB | reference F1 "
                    f"{ref['f1'] * 100:.2f} (TP {ref['TP']} FP {ref['FP']} TN {ref['TN']} FN {ref['FN']}) -> "
                    f"{'MATCH' if info['ref_match'] else 'MISMATCH'}" + ("" if not ARGS.smoke else " (smoke: not comparable)"))
            else:
                info["status"] = "error"
                with open(errors_path, "a") as f:
                    f.write(json.dumps(info) + "\n")
                log(f"[{n_done}/{len(pending)}] run {info['run']} {info['config']} FAILED: {info['error']}")
            details = pd.DataFrame([{k: v for k, v in info.items() if k != "traceback"}])
            details.to_csv(details_path, mode="a", header=not os.path.exists(details_path), index=False)

    assemble_predictions(out_dir, ARGS.runs)
    if not ARGS.smoke:
        compare_with_reference(reference, results_path, out_dir)
    log(f"Finished in {(time.time() - started) / 60:.1f} min; free disk {free_gb(out_dir):.1f} GB; outputs in {out_dir}")


if __name__ == "__main__":
    main()
