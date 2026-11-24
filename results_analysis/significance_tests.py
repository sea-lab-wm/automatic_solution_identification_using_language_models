"""Statistical significance of the main comparative claims (EMSE revision, R3 / Editor).

Reviewer 3 and the editor asked whether the differences behind the main claims
(e.g. Llama-3ft vs RoBERTa, best ensemble vs Llama-3ft) are statistically significant.
For every main claim of the paper (findings boxes, abstract, intro, conclusions) we run
two paired tests on the saved predictions (no model is trained):

  A. Issue-level paired bootstrap: resample the issues with replacement (all comments
     of a sampled issue together), B = 10,000, fixed seed. For each resample, compute
     metric(A) - metric(B) on the pooled comments of the resample (same comments for
     both models). Reports the observed difference, the 95% percentile CI and the
     two-sided p = 2 * min(P(d <= 0), P(d >= 0)), floored at 1/B.
     Needs per-comment predictions.
  B. Fold-level Wilcoxon signed-rank test on the per-fold metric values (paired by
     fold; 10 CV folds, 9 temporal folds), two-sided, exact, with the rank-biserial
     correlation (W+ - W-) / (W+ + W-) as effect size (zero differences are dropped).

Holm correction is applied within each family (RQ1, RQ2, RQ3, RQ4, Temporal) and metric,
separately for each test type, over all claim x metric tests of the family.

Metrics are pooled over the comments (as in the paper's tables), not averaged over
folds. Prompting: the paper averages the counts of the 3 executions of each prompt
(Table 5), so the TP/FP/FN counts of the 3 executions are summed before computing the
metrics, both in the bootstrap and per fold.

Data:
  - results_analysis/ensembled_models.xlsx: per-comment CV predictions of the 12 best
    configurations and the best ensemble ("Best 3 from each category").
  - prompting/generated_responses_test_set/fold*/responses-<id>.csv: prompted Llama-3,
    3 executions per prompt (non-0/1 answers count as non-solution).
  - results/rerun-ml-table4/predictions/ml/run_<fold>_with_confidence.csv: per-comment
    predictions of all 48 ML configurations (re-run of Table 4; 456 of the 480
    configuration-fold runs reproduce the original TP/FP/TN/FN exactly, the rest differ
    by a few comments, see results/rerun-ml-table4/ml/reference_comparison.csv). Used
    for the embedding claims of RQ1, which now get both tests.
  - results/ml/comment_data_results copy.csv: original per-fold TP/FP/TN/FN of the ML
    configurations (only for the sanity check of the re-run).
  - results/predictions/temporal-prediction/temporal_vs_old_predictions.csv: temporal and
    CV predictions of SVM, RoBERTa, Llama-3ft; the 4,408 comments in the test sets of both
    settings are the rows with in_old = True.

Usage: python results_analysis/significance_tests.py
Requires: pandas, numpy, scipy, openpyxl
"""

import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

PREDICTIONS_PATH = "results_analysis/ensembled_models.xlsx"
SPLIT_PATH = "dataset/solution_identification_data/train_test_split.json"
PROMPT_DIR = "prompting/generated_responses_test_set"
ML_FOLDS_PATH = "results/ml/comment_data_results copy.csv"
ML_RERUN_DIR = "results/rerun-ml-table4/predictions/ml"
TEMPORAL_PATH = "results/predictions/temporal-prediction/temporal_vs_old_predictions.csv"
OUTPUT_DIR = "results_analysis/generated/significance_tests"

KEY = ["issue_id", "text_id"]
B = 10_000
SEED = 20251029
ALPHA = 0.05
METRICS = ["precision", "recall", "f1"]
PR, RE, F1 = ["precision"], ["recall"], ["f1"]
PRF = METRICS

# Display names -> source columns. "prompt:<id>" = prompted Llama-3 (3 executions).
ENSEMBLE = "Best 3 from each category"
PROMPTS = {"Prompt ZS (no desc.)": "1.1.0.3", "Prompt ZS (desc.)": "1.1.1.3",
           "Prompt FS-7 (no desc.)": "1.2.0.4", "Prompt FS-7 (desc.)": "1.2.1.4",
           "Prompt CoT-7 (no desc.)": "1.3.0.4", "Prompt CoT-7 (desc.)": "1.3.1.4"}
CV_MODELS = {"Llama-3ft": "LLAMA-FT", "RoBERTa": "RoBERTa", "BERT": "BERT", "XLNet": "XLNet",
             "DistilBERT": "DistilBERT", "SVM": "SVC", "LR": "Logistic_Regression",
             "Ensemble (SVM+RoBERTa+Llama-3ft)": ENSEMBLE}
ML_NAMES = {"SVC": "SVM", "Logistic_Regression": "LR", "Random_Forest": "RF",
            "Decision_Tree": "DT", "K_Nearest_Neighbors": "KNN", "Naive_Bayes": "NB"}
REPRESENTATIONS = {"tfidf": ("all", "tfidf"), "bert": ("none", "bert"),
                   "llama": ("none", "llama"), "gpt": ("none", "gpt")}

# Claim inventory: (id, family, claim, quote, location, model A, model B, metrics, source)
# source: "cv" = per-comment CV predictions, "ml" = fold-level ML results only,
# "temporal" = temporal predictions, "drop" = CV vs temporal predictions of one model.
Q_EMB = ("\\mlms with LLM embeddings outperform \\embtfidf features [abstract]; "
         "all the ML models show higher F1 performance when using any LM embeddings, regardless of data "
         "balancing, with the exception of \\dtrees and \\randforests [4_results.tex:410]")
L_EMB = "0_main.tex:145; 1_intro.tex:43; 4_results.tex:410; 7_conclusions.tex:10"
CLAIMS = [
    *[(f"RQ1-emb-{short}", "RQ1", f"Llama embeddings vs TF-IDF for {short} (no DB)", Q_EMB, L_EMB,
       f"{short} Llama emb.", f"{short} TF-IDF", F1, "ml") for short in ML_NAMES.values()],
    ("RQ1-svm-gpt", "RQ1", "SVM: GPT embeddings vs TF-IDF (no DB)",
     "the best-performing \\mlm, \\svcs, improved its F1 score by 41.6\\% using \\embgpt over \\embtfidf",
     "4_results.tex:475; 1_intro.tex:43", "SVM GPT emb.", "SVM TF-IDF", F1, "ml"),
    ("RQ1-avg-tfidf", "RQ1", "Llama embeddings vs TF-IDF, average F1 over the 6 ML models (no DB)",
     "\\embllama achieves the highest F1 performance on average (0.508 F1 with no balancing ...), "
     "outperforming \\embtfidf's scores (0.427 F1 with no balancing ...)",
     "4_results.tex:422", "Avg-6 Llama emb.", "Avg-6 TF-IDF", F1, "ml"),
    ("RQ1-avg-gpt", "RQ1", "Llama embeddings vs GPT (2nd best), average F1 over the 6 ML models (no DB)",
     "Among the embeddings, \\embllama produces the most consistent improvements, yielding the highest "
     "average F1 across models", "4_results.tex:485 (findings); 4_results.tex:422",
     "Avg-6 Llama emb.", "Avg-6 GPT emb.", F1, "ml"),
    ("RQ1-svm-lr", "RQ1", "Best ML model (SVM, GPT) vs 2nd best (LR, GPT)",
     "The overall best model, \\svcs (with \\embgpt and no data balancing), achieved the highest F1 score "
     "of 0.663, outperforming the second best model by 4.02\\% (\\lrs with \\embgpt and no data balancing)",
     "4_results.tex:473; 7_conclusions.tex:10", "SVM", "LR", PRF, "cv"),
    ("RQ2-cot-zs", "RQ2", "Prompting strategies similar: CoT-7 vs ZS (best prompts, no desc.)",
     "These prompts show comparable performance, with CoT performing the best [752]; "
     "all strategies yield similar results [760]", "4_results.tex:752; 4_results.tex:760",
     "Prompt CoT-7 (no desc.)", "Prompt ZS (no desc.)", PRF, "cv"),
    ("RQ2-cot-fs", "RQ2", "Prompting strategies similar: CoT-7 vs FS-7 (best prompts, no desc.)",
     "These prompts show comparable performance, with CoT performing the best [752]; "
     "all strategies yield similar results [760]", "4_results.tex:752; 4_results.tex:760",
     "Prompt CoT-7 (no desc.)", "Prompt FS-7 (no desc.)", PRF, "cv"),
    *[(f"RQ2-desc-{s}", "RQ2", f"{s}: with vs without issue description",
       "Including the issue description reduces the F1 score by 6\\% (ZS), 6.8\\% (FS), and 8.5\\% (CoT), "
       "primarily due to significant precision degradation (8.7\\% to 12.4\\%) that outweighs the recall "
       "improvement [755]; Including issue descriptions further reduces precision [808, findings]",
       "4_results.tex:755; 4_results.tex:808", f"Prompt {s} (desc.)", f"Prompt {s} (no desc.)", PRF, "cv")
      for s in ["ZS", "FS-7", "CoT-7"]],
    ("RQ2-prompt-svm", "RQ2", "Best prompt (CoT-7) vs best ML model (SVM)",
     "\\llama's relatively low performance via prompting (compared to the best \\mlms ...) [760]; "
     "prompting underperforms [abstract]; Even with its best prompting strategy (chain-of-thought), "
     "\\llama underperformed [conclusions]", "4_results.tex:760; 0_main.tex:145; 7_conclusions.tex:12",
     "Prompt CoT-7 (no desc.)", "SVM", PRF, "cv"),
    *[(f"RQ3-rob-{m.lower()}", "RQ3", f"Best PLM (RoBERTa) vs {m}",
       "\\rbert achieved the highest F1 scores for both imbalanced (0.689) and balanced (0.692) data, with a "
       "maximum relative improvement of 5.8\\% over \\dbert with data balancing (the 2nd best model) [869]; "
       "\\rbert achieves the best overall performance among the PLMs [921, findings]",
       "4_results.tex:869; 4_results.tex:921", "RoBERTa", m, F1, "cv")
      for m in ["BERT", "XLNet", "DistilBERT"]],
    ("RQ3-llama-rob", "RQ3", "Llama-3ft vs RoBERTa (Reviewer 3 example)",
     "Compared to the best PLM (\\ie\\ \\rbert), \\llamaft attains substantially higher recall with only a "
     "slight precision drop, resulting in a clear F1 gain [921, findings]; ... resulting in a 3.5\\% F1 "
     "increase [912]", "4_results.tex:921; 4_results.tex:912; 7_conclusions.tex:12",
     "Llama-3ft", "RoBERTa", PRF, "cv"),
    ("RQ3-llama-svm", "RQ3", "Llama-3ft vs best ML model (SVM)",
     "\\llamaft achieved the highest F1 score of 0.716, outperforming all PLMs and \\mlms [912]; "
     "\\llamaft surpasses all models [921, findings]", "4_results.tex:912; 4_results.tex:921",
     "Llama-3ft", "SVM", PRF, "cv"),
    ("RQ3-llama-prompt", "RQ3", "Fine-tuned vs prompted Llama-3 (best prompt CoT-7)",
     "The fine-tuned \\llama (\\ie~\\llamaft) and \\rbert models ... substantially outperforming both "
     "traditional \\mlms and prompted LLMs", "1_intro.tex:43; 7_implications.tex:110; 0_main.tex:145",
     "Llama-3ft", "Prompt CoT-7 (no desc.)", PRF, "cv"),
    ("RQ3-rob-svm", "RQ3", "RoBERTa vs best ML model (SVM)",
     "The fine-tuned \\llama (\\ie~\\llamaft) and \\rbert models ... substantially outperforming both "
     "traditional \\mlms and prompted LLMs", "1_intro.tex:43; 7_implications.tex:110",
     "RoBERTa", "SVM", F1, "cv"),
    ("RQ3-rob-prompt", "RQ3", "RoBERTa vs prompted Llama-3 (best prompt CoT-7)",
     "The fine-tuned \\llama (\\ie~\\llamaft) and \\rbert models ... substantially outperforming both "
     "traditional \\mlms and prompted LLMs", "1_intro.tex:43; 7_implications.tex:110",
     "RoBERTa", "Prompt CoT-7 (no desc.)", F1, "cv"),
    ("RQ4-ens-llama", "RQ4", "Best ensemble (SVM+RoBERTa+Llama-3ft) vs Llama-3ft (Reviewer 3 example)",
     "Combining different models leads to modest performance improvements, with the combination of the best "
     "models from each type (\\svc, \\rbert, and \\llamaft) achieving the highest F1 gain of 2.9\\% compared "
     "to \\llamaft [findings]; Ensembles of the best models further improve results (0.737 F1) [abstract]",
     "4_results.tex:1110; 4_results.tex:1098; 0_main.tex:145; 1_intro.tex:60",
     "Ensemble (SVM+RoBERTa+Llama-3ft)", "Llama-3ft", PRF, "cv"),
    ("TMP-rob-llama", "Temporal", "Temporal: RoBERTa vs Llama-3ft",
     "However, \\rbert outperforms \\llamaft in this setting (F1 of 0.633 vs. 0.572), especially when "
     "little training data is available.", "4_results.tex:1391 (findings); 4_results.tex:1388",
     "RoBERTa", "Llama-3ft", PRF, "temporal"),
    *[(f"TMP-{m.split('-')[0].lower()}-svm", "Temporal", f"Temporal: {m} vs SVM",
       "The fine-tuned LMs still outperform \\svcs.", "4_results.tex:1391 (findings); 4_results.tex:1388",
       m, "SVM", F1, "temporal") for m in ["RoBERTa", "Llama-3ft"]],
    *[(f"TMP-drop-{m.split('-')[0].lower()}", "Temporal", f"{m}: temporal vs 10-fold CV (same 4,408 comments)",
       "All three best model's performance decreased when they are trained on earlier comments and tested on "
       "later ones (F1 drops of 0.059 to 0.140)", "4_results.tex:1391 (findings); 4_results.tex:1382",
       f"{m} temporal", f"{m} CV", F1, "drop") for m in ["SVM", "RoBERTa", "Llama-3ft"]],
]
# Test precision, recall, and F1 for every comparison, so all three are reported (and Holm-corrected) consistently.
# The ML configuration claims ("ml") now use the per-comment predictions of the re-run (bootstrap + Wilcoxon).
CLAIMS = [(*c[:7], PRF, "cv" if c[8] == "ml" else c[8]) for c in CLAIMS]
TEMPORAL_COLS = {"SVM": "SVC", "RoBERTa": "RoBERTa", "Llama-3ft": "LLAMA-FT"}


# ---------------------------------------------------------------- metrics

def prf(tp, fp, fn):
    """Precision, recall, F1 from (arrays of) counts; 0 where undefined."""
    tp, fp, fn = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return {"precision": p, "recall": r, "f1": f}


def comment_counts(y, pred):
    """Per-comment TP/FP/FN indicators (n x 3)."""
    y, pred = np.asarray(y), np.asarray(pred)
    return np.stack([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)], axis=1).astype(float)


# ---------------------------------------------------------------- data

def load_cv():
    """Per-comment CV predictions: label, fold, one TP/FP/FN matrix per model name."""
    x = pd.read_excel(PREDICTIONS_PATH, sheet_name=0, header=1)
    x.columns = [str(c).strip() for c in x.columns]
    x = x[pd.to_numeric(x["label"], errors="coerce").isin([0, 1])
          & pd.to_numeric(x["issue_id"], errors="coerce").notna()].copy()
    x[KEY] = x[KEY].astype(int)
    x["label"] = x["label"].astype(int)
    split = json.load(open(SPLIT_PATH))
    folds = pd.DataFrame([{**r, "fold": i} for i, run in enumerate(split) for r in split[run]["test"]])
    x = x.merge(folds, on=KEY, how="left", validate="one_to_one")
    assert x["fold"].notna().all() and len(x) == 4867, len(x)
    counts = {name: comment_counts(x["label"], x[col].astype(int)) for name, col in CV_MODELS.items()}
    for name, pid in PROMPTS.items():
        d = pd.concat([pd.read_csv(f"{PROMPT_DIR}/fold{f}/responses-{pid}.csv") for f in range(10)])
        d[KEY] = d[KEY].astype(int)
        m = x[KEY + ["label"]].merge(d, on=KEY, how="left", suffixes=("", "_p"), validate="one_to_one")
        assert m["label_p"].notna().all() and (m["label"] == m["label_p"]).all(), pid
        # sum of the 3 executions = 3 x the averaged counts used in Table 5 (ratios are identical)
        counts[name] = sum(comment_counts(m["label"], pd.to_numeric(m[f"llama_prediction_{e}"], errors="coerce")
                                          .eq(1).astype(int)) for e in range(3))
    # all ML configurations (no data balancing) from the re-run of Table 4
    for model, short in ML_NAMES.items():
        for rep_, (pre, emb) in REPRESENTATIONS.items():
            col = lambda f: f"{pre}_{emb}_nos_{model}_{f}"
            d = pd.concat([pd.read_csv(f"{ML_RERUN_DIR}/run_{f}_with_confidence.csv", usecols=KEY + ["label", col(f)])
                           .rename(columns={col(f): "pred"}) for f in range(10)])
            d[KEY] = d[KEY].astype(int)
            m = x[KEY + ["label"]].merge(d, on=KEY, how="left", suffixes=("", "_r"), validate="one_to_one")
            assert m["pred"].notna().all() and (m["label"] == m["label_r"]).all(), (model, rep_)
            counts[ml_name(short, rep_)] = comment_counts(m["label"], m["pred"].astype(int))
    return x[["issue_id", "fold"]].reset_index(drop=True), counts


def ml_name(short, rep):
    return f"{short} {'TF-IDF' if rep == 'tfidf' else rep.upper() if rep != 'llama' else 'Llama'}" + \
        ("" if rep == "tfidf" else " emb.")


# "Avg-6 <representation>": mean of the metric over the six ML models (as in the paper's average row)
AVG6 = {f"Avg-6 {label}": [ml_name(s, rep) for s in ML_NAMES.values()]
        for rep, label in [("tfidf", "TF-IDF"), ("llama", "Llama emb."), ("gpt", "GPT emb."), ("bert", "BERT emb.")]}


def load_temporal():
    t = pd.read_csv(TEMPORAL_PATH)
    t = t[t["in_old"].astype(bool)].copy()  # the 4,408 comments in the test sets of both settings
    for c in TEMPORAL_COLS.values():
        t[f"{c}_temporal"] = t[f"temporal_{c}"].astype(int)
        t[f"{c}_old"] = t[f"old_{c}"].astype(int)
    assert len(t) == 4408 and not t.duplicated(KEY).any()
    counts = {}
    for name, col in TEMPORAL_COLS.items():
        counts[f"{name} temporal"] = comment_counts(t["label"], t[f"{col}_temporal"])
        counts[f"{name} CV"] = comment_counts(t["label"], t[f"{col}_old"])
    counts.update({n: counts[f"{n} temporal"] for n in TEMPORAL_COLS})
    return t[["issue_id", "fold"]].reset_index(drop=True), counts


def load_ml_folds():
    """Per-fold metrics (no DB) for each ML model x representation, plus the 6-model average."""
    d = pd.read_csv(ML_FOLDS_PATH)
    d = d[~d["oversampling"]]
    per_fold, pooled = {}, {}
    for model, short in ML_NAMES.items():
        for rep, (pre, emb) in REPRESENTATIONS.items():
            g = d[(d["model"] == model) & (d["preprocess"] == pre) & (d["embedding"] == emb)].sort_values("run")
            assert list(g["run"]) == list(range(10)), (model, rep)
            name = f"{short} {'TF-IDF' if rep == 'tfidf' else rep.upper() if rep != 'llama' else 'Llama'}"
            name += "" if rep == "tfidf" else " emb."
            per_fold[name] = prf(g["TP"], g["FP"], g["FN"])
            pooled[name] = {k: float(v) for k, v in prf(g["TP"].sum(), g["FP"].sum(), g["FN"].sum()).items()}
    for rep, label in [("tfidf", "TF-IDF"), ("llama", "Llama emb."), ("gpt", "GPT emb."), ("bert", "BERT emb.")]:
        members = [n for n in per_fold if n.split(" ", 1)[1] == label]
        assert len(members) == 6
        name = f"Avg-6 {label}"
        # the paper's "average" row is the mean of the 6 pooled per-model scores
        per_fold[name] = {k: np.mean([per_fold[n][k] for n in members], axis=0) for k in METRICS}
        pooled[name] = {k: float(np.mean([pooled[n][k] for n in members])) for k in METRICS}
    return per_fold, pooled


# ---------------------------------------------------------------- tests

def issue_matrix(meta, counts):
    """Aggregate per-comment counts per issue: dict name -> (n_issues x 3)."""
    codes, issues = pd.factorize(meta["issue_id"])
    out = {}
    for name, c in counts.items():
        agg = np.zeros((len(issues), 3))
        np.add.at(agg, codes, c)
        out[name] = agg
    return out


def bootstrap_weights(n_issues, rng):
    """B x n_issues matrix: how often each issue is drawn in each resample."""
    idx = rng.integers(0, n_issues, size=(B, n_issues))
    w = np.zeros((B, n_issues), dtype=np.int32)
    np.add.at(w, (np.repeat(np.arange(B), n_issues), idx.ravel()), 1)
    return w


def resampled_metric(per_issue, weights, name, metric):
    """Metric of a model (or average over the six ML models) in each bootstrap resample."""
    if name in AVG6:
        return np.mean([resampled_metric(per_issue, weights, m, metric) for m in AVG6[name]], axis=0)
    return prf(*(weights @ per_issue[name]).T)[metric]


def pooled_metric(counts, name, metric):
    if name in AVG6:
        return float(np.mean([pooled_metric(counts, m, metric) for m in AVG6[name]]))
    return float(prf(*counts[name].sum(0))[metric])


def bootstrap(per_issue, weights, a, b, metric):
    d = resampled_metric(per_issue, weights, a, metric) - resampled_metric(per_issue, weights, b, metric)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return lo, hi, float(min(1.0, max(p, 1.0 / B)))


def fold_metrics(meta, counts, name):
    if name in AVG6:
        return {k: np.mean([fold_metrics(meta, counts, m)[k] for m in AVG6[name]], axis=0) for k in METRICS}
    agg = pd.DataFrame(counts[name], columns=["tp", "fp", "fn"]).groupby(meta["fold"].values).sum().sort_index()
    return prf(agg["tp"], agg["fp"], agg["fn"])


def exact_signflip_p(d):
    """Exact two-sided Wilcoxon p by enumerating all sign assignments (midranks; handles ties)."""
    ranks = stats.rankdata(np.abs(d))
    obs = ranks[d > 0].sum()
    t = np.array([ranks[list(s)].sum() for k in range(len(d) + 1) for s in itertools.combinations(range(len(d)), k)])
    lo, hi = np.mean(t <= obs + 1e-9), np.mean(t >= obs - 1e-9)
    return min(1.0, 2 * min(lo, hi))


def wilcoxon(xa, xb):
    d = np.asarray(xa) - np.asarray(xb)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0, 0.0, 0, "all differences zero"
    ranks = stats.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    rbc = (wp - wm) / (wp + wm)
    if len(np.unique(np.abs(d))) == len(d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = stats.wilcoxon(d, method="exact").pvalue
        note = ""
    else:
        p, note = exact_signflip_p(d), "ties: exact sign-flip enumeration with midranks"
    return float(p), float(rbc), len(d), note


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate(pv[order] * (len(pv) - np.arange(len(pv)))))
    res = np.empty(len(pv))
    res[order] = adj
    out[ok] = res
    return out


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    cv_meta, cv_counts = load_cv()
    tmp_meta, tmp_counts = load_temporal()
    ml_folds, ml_pooled = load_ml_folds()
    cv_issue, tmp_issue = issue_matrix(cv_meta, cv_counts), issue_matrix(tmp_meta, tmp_counts)
    cv_w = bootstrap_weights(len(next(iter(cv_issue.values()))), rng)
    tmp_w = bootstrap_weights(len(next(iter(tmp_issue.values()))), rng)

    # sanity checks against the paper (pooled metrics, 3 decimals)
    expected = {"Llama-3ft": (0.626, 0.838, 0.716), "RoBERTa": (0.673, 0.712, 0.692), "SVM": (0.629, 0.701, 0.663),
                "Ensemble (SVM+RoBERTa+Llama-3ft)": (0.702, 0.775, 0.737), "LR": (0.550, 0.758, 0.637),
                "BERT": (0.690, 0.648, 0.668), "XLNet": (0.616, 0.713, 0.661), "DistilBERT": (0.681, 0.641, 0.660),
                "Prompt CoT-7 (no desc.)": (0.414, 0.686, 0.516), "Prompt ZS (no desc.)": (0.406, 0.700, 0.514),
                "Prompt FS-7 (no desc.)": (0.394, 0.729, 0.511), "Prompt ZS (desc.)": (0.356, 0.753, 0.483),
                "Prompt FS-7 (desc.)": (0.344, 0.774, 0.476), "Prompt CoT-7 (desc.)": (0.378, 0.632, 0.473)}
    for name, exp in expected.items():
        got = tuple(round(float(v), 3) for v in prf(*cv_counts[name].sum(0)).values())
        print(f"check {name:34s} {got} {'OK' if got == exp else 'MISMATCH vs ' + str(exp)}")
    tmp_expected = {"SVM temporal": (0.524, 0.589, 0.555), "RoBERTa temporal": (0.612, 0.656, 0.633),
                    "Llama-3ft temporal": (0.488, 0.690, 0.572), "SVM CV": (0.608, 0.683, 0.643),
                    "RoBERTa CV": (0.671, 0.714, 0.692), "Llama-3ft CV": (0.620, 0.838, 0.712)}
    for name, exp in tmp_expected.items():
        got = tuple(round(float(v), 3) for v in prf(*tmp_counts[name].sum(0)).values())
        print(f"check {name:34s} {got} {'OK' if got == exp else 'MISMATCH vs ' + str(exp)}")
    paper_fold_f1 = {"SVM": [0.575, 0.474, 0.641, 0.636, 0.576, 0.511, 0.526, 0.560, 0.444],
                     "RoBERTa": [0.660, 0.629, 0.632, 0.614, 0.671, 0.582, 0.657, 0.662, 0.568],
                     "Llama-3ft": [0.418, 0.582, 0.535, 0.607, 0.694, 0.514, 0.622, 0.695, 0.596]}
    for name, exp in paper_fold_f1.items():
        got = [round(float(v), 3) for v in fold_metrics(tmp_meta, tmp_counts, name)["f1"]]
        print(f"check temporal folds {name:20s} {'OK' if got == exp else 'MISMATCH ' + str(got)}")
    for name, exp in {"Avg-6 Llama emb.": 0.508, "Avg-6 TF-IDF": 0.427, "Avg-6 GPT emb.": 0.457}.items():
        got = round(ml_pooled[name]["f1"], 3)
        print(f"check {name:34s} F1 {got} {'OK' if got == exp else 'MISMATCH vs ' + str(exp)}")

    for name in [ml_name(s, r) for s in ML_NAMES.values() for r in ("tfidf", "llama", "gpt")] + \
            ["Avg-6 TF-IDF", "Avg-6 Llama emb.", "Avg-6 GPT emb."]:
        got = round(pooled_metric(cv_counts, name, "f1"), 3)
        ref = round(ml_pooled[name]["f1"], 3)
        print(f"check re-run {name:28s} F1 {got} vs original {ref} {'OK' if got == ref else 'DIFF'}")

    claims_rows, rows = [], []
    for cid, fam, claim, quote, loc, a, b, metrics, src in CLAIMS:
        tests = {"cv": "bootstrap + Wilcoxon (10 CV folds)", "ml": "Wilcoxon only (10 CV folds; no per-comment "
                 "predictions for this configuration)", "temporal": "bootstrap + Wilcoxon (9 temporal folds)",
                 "drop": "bootstrap + Wilcoxon (9 temporal folds, CV predictions restricted to each fold)"}[src]
        claims_rows.append({"claim_id": cid, "family": fam, "claim": claim, "model_a": a, "model_b": b,
                            "metrics": "/".join(metrics), "tests": tests, "location": loc, "quote": quote})
        for metric in metrics:
            r = {"claim_id": cid, "family": fam, "claim": claim, "model_a": a, "model_b": b, "metric": metric}
            if src == "ml":
                va, vb = ml_pooled[a][metric], ml_pooled[b][metric]
                fa, fb = ml_folds[a][metric], ml_folds[b][metric]
                lo = hi = pb = np.nan
            else:
                meta, cnt, per_issue, w = ((cv_meta, cv_counts, cv_issue, cv_w) if src == "cv"
                                           else (tmp_meta, tmp_counts, tmp_issue, tmp_w))
                va, vb = (pooled_metric(cnt, m, metric) for m in (a, b))
                lo, hi, pb = bootstrap(per_issue, w, a, b, metric)
                fa, fb = fold_metrics(meta, cnt, a)[metric], fold_metrics(meta, cnt, b)[metric]
            pw, rbc, n, note = wilcoxon(fa, fb)
            r.update({"value_a": va, "value_b": vb, "delta": va - vb, "boot_ci_low": lo, "boot_ci_high": hi,
                      "boot_p": pb, "wilcoxon_p": pw, "rank_biserial": rbc, "n_folds": len(fa),
                      "folds_a_better": int(np.sum(np.asarray(fa) > np.asarray(fb))),
                      "n_nonzero_fold_diffs": n, "note": note})
            rows.append(r)

    res = pd.DataFrame(rows)
    # Holm correction within each RQ and metric (precision, recall, and F1 are separate questions)
    for fam, idx in res.groupby(["family", "metric"]).groups.items():
        res.loc[idx, "boot_p_holm"] = holm(res.loc[idx, "boot_p"])
        res.loc[idx, "wilcoxon_p_holm"] = holm(res.loc[idx, "wilcoxon_p"])
    res["boot_sig"] = np.where(res["boot_p"].isna(), "", np.where(res["boot_p_holm"] < ALPHA, "yes", "no"))
    res["wilcoxon_sig"] = np.where(res["wilcoxon_p_holm"] < ALPHA, "yes", "no")
    cols = ["claim_id", "family", "claim", "model_a", "model_b", "metric", "value_a", "value_b", "delta",
            "boot_ci_low", "boot_ci_high", "boot_p", "boot_p_holm", "boot_sig", "wilcoxon_p", "wilcoxon_p_holm",
            "wilcoxon_sig", "rank_biserial", "n_folds", "folds_a_better", "n_nonzero_fold_diffs", "note"]
    res = res[cols]
    res.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)  # full precision; round only when reporting
    pd.DataFrame(claims_rows).to_csv(f"{OUTPUT_DIR}/claims.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
    show = res[["claim_id", "metric", "value_a", "value_b", "delta", "boot_ci_low", "boot_ci_high", "boot_p_holm",
                "wilcoxon_p_holm", "rank_biserial", "folds_a_better", "note"]]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved {OUTPUT_DIR}/results.csv and claims.csv")


if __name__ == "__main__":
    main()
