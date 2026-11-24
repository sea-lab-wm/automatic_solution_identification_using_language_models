"""Analyses that explain the main findings (EMSE revision, R2.C5).

All analyses are deterministic: they use the dataset, the saved per-comment
predictions of the paper's runs (for prompting: the three executions of the
best prompt, averaged, as in the paper), and the saved tuning results. No model is trained.

1a. Words that best distinguish solution from non-solution comments (chi-square, all words).
1b. Recall per kind of solution comment (solution code).
2.  Class imbalance and data balancing: class weights chosen by tuning, and the
    effect of SMOTE on precision/recall per ML model.
3.  Prompted Llama-3: precision/recall, and false positives on non-solution
    comments with vs. without technical content (code identifiers, file paths).
4.  Error overlap among the three best models (SVM, RoBERTa, Llama-3_ft).
5.  Problem categories: share of code descriptions among solution comments vs. F1.

Usage: python results_analysis/explain_findings.py
Requires: pandas, openpyxl, scipy, scikit-learn
"""

import os
import re

import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2


COMMENT_DATA_PATH = "dataset/solution_identification_data/labeled_comment_data.csv"
PREDICTIONS_PATH = "results_analysis/ensembled_models.xlsx"
ML_TUNING_PATH = "results/ml/comment_data_results copy.csv"
# Best prompt in the paper (CoT, seven examples, no issue description); three executions per comment
PROMPT_RESPONSES = "prompting/generated_responses_test_set/fold{}/responses-1.3.0.4.csv"
OUTPUT_DIR = "results_analysis/generated/explain_findings"

CODE_NAMES = {"CODE_IMPLEMENTATION": "CODE_DESCRIPTION", "POTENTIAL_SOLUTION_DESIGN": "PROPOSED_SOLUTION",
              "SOLUTION_REVIEW": "SOLUTION_REVIEW"}
BEST = {"SVM": "SVC", "RoBERTa": "RoBERTa", "Llama-3_ft": "LLAMA-FT"}
PROMPT_RUNS = ["llama_prediction_0", "llama_prediction_1", "llama_prediction_2"]
PAPER_REPRESENTATIONS = [("all", "tfidf"), ("none", "bert"), ("none", "llama"), ("none", "gpt")]
CLASS_WEIGHT_MODELS = ["Logistic_Regression", "Decision_Tree", "Random_Forest", "SVC"]

# Technical content: code identifiers (camelCase, snake_case, calls), file names with extensions, C++ scopes
TECHNICAL = re.compile(r"\b[a-z]+[A-Z]\w*\b|\b\w+_\w+\b|\b\w+\(\)|\b[\w/.-]+\.(?:cpp|h|js|jsm|py|java|xul|css|html|rs|c|mm|idl|xml|json)\b|::")


def out(name):
    return os.path.join(OUTPUT_DIR, name)


def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def load_predictions():
    x = pd.read_excel(PREDICTIONS_PATH, sheet_name=0, header=1)
    x.columns = [str(c).strip() for c in x.columns]
    x = x[pd.to_numeric(x["label"], errors="coerce").isin([0, 1]) & pd.to_numeric(x["issue_id"], errors="coerce").notna()].copy()
    x["label"] = x["label"].astype(int)
    for col in BEST.values():
        x[col] = x[col].astype(int)
    x["issue_category"] = x["issue_category"].astype(str).str.strip().str.lower()
    return x


def load_prompt_runs():
    d = pd.concat([pd.read_csv(PROMPT_RESPONSES.format(f)) for f in range(10)], ignore_index=True)
    for col in PROMPT_RUNS:
        d[col] = (pd.to_numeric(d[col], errors="coerce") == 1).astype(int)  # non-binary responses count as non-solution
    return d


def mean_over_runs(d, fn):
    vals = [fn(d[col]) for col in PROMPT_RUNS]
    return tuple(round(sum(v[k] for v in vals) / len(vals), 3) for k in range(len(vals[0])))


def analysis_1a(comments):
    vec = CountVectorizer(lowercase=True, stop_words="english", min_df=1, binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(comments["text"].fillna(""))
    y = comments["label"].values
    scores, _ = chi2(X, y)
    words = vec.get_feature_names_out()
    sol = X[y == 1].mean(axis=0).A1; non = X[y == 0].mean(axis=0).A1
    df = pd.DataFrame({"word": words, "chi2": scores, "pct_solution_comments": 100 * sol, "pct_non_solution_comments": 100 * non})
    df = df[df.pct_solution_comments > df.pct_non_solution_comments].sort_values("chi2", ascending=False).head(15)
    return df.round(1)


def analysis_1a_phrase(comments):
    """Share of solution and non-solution comments that contain the text "Created attachment",
    which Bugzilla adds to a comment when a patch is uploaded."""
    has = comments["text"].fillna("").str.contains("Created attachment", case=False, regex=False)
    sol = comments["label"] == 1
    return {"pct_solution_comments": round(100 * has[sol].mean(), 1),
            "pct_non_solution_comments": round(100 * has[~sol].mean(), 1)}


def analysis_1b(pred, prompt):
    sol = pred[pred.label == 1].copy()
    sol["kind"] = sol["code"].map(CODE_NAMES)
    psol = prompt[prompt.label == 1].copy()
    psol["kind"] = psol["code"].map(CODE_NAMES)
    rows = []
    for kind, g in sol.groupby("kind"):
        row = {"kind": kind, "n": len(g)}
        for name, col in BEST.items():
            row[name] = round(g[col].mean(), 3)
        row["Llama-3 (prompted)"] = mean_over_runs(psol[psol.kind == kind], lambda c: (c.mean(),))[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def analysis_2(comments):
    tuning = pd.read_csv(ML_TUNING_PATH)
    tuning = tuning[tuning.model != "model"].copy()
    tuning["rep"] = list(zip(tuning.preprocess, tuning.embedding))
    tuning = tuning[tuning.rep.isin(PAPER_REPRESENTATIONS)]
    for c in ["TP", "FP", "FN"]:
        tuning[c] = tuning[c].astype(float)
    nobal = tuning[(tuning.oversampling.astype(str) == "False") & tuning.model.isin(CLASS_WEIGHT_MODELS)]
    weighted = int(nobal.param.astype(str).str.contains("'class_weight': 'balanced'").sum())
    rows = []
    for model, g in tuning.groupby("model"):
        for bal, h in g.groupby(g.oversampling.astype(str)):
            tp, fp, fn = h.TP.sum(), h.FP.sum(), h.FN.sum()
            p = tp / (tp + fp); r = tp / (tp + fn)
            rows.append({"model": model, "smote": bal == "True", "precision": p, "recall": r, "f1": 2 * p * r / (p + r)})
    d = pd.DataFrame(rows)
    eff = d.pivot(index="model", columns="smote", values=["precision", "recall", "f1"])
    effect = pd.DataFrame({
        "class_weights_available": [m in CLASS_WEIGHT_MODELS for m in eff.index],
        "precision_no_smote": eff[("precision", False)], "precision_smote": eff[("precision", True)],
        "recall_no_smote": eff[("recall", False)], "recall_smote": eff[("recall", True)],
        "f1_no_smote": eff[("f1", False)], "f1_smote": eff[("f1", True)],
    }, index=eff.index).round(3)
    n_sol = int((comments.label == 1).sum()); n_non = int((comments.label == 0).sum())
    summary = {"solution_comments": n_sol, "non_solution_comments": n_non, "ratio": round(n_non / n_sol, 1),
               "class_weight_runs": weighted, "runs_without_smote_for_lr_dt_rf_svm": len(nobal)}
    return summary, effect


def analysis_3(pred, prompt):
    non = pred[pred.label == 0].copy()
    non["technical"] = non["text"].astype(str).str.contains(TECHNICAL)
    pnon = prompt[prompt.label == 0].copy()
    pnon["technical"] = pnon["text"].astype(str).str.contains(TECHNICAL)
    p, r, f1 = mean_over_runs(prompt, lambda c: prf(prompt.label, c))
    rows = [{"model": "Llama-3 (prompted)", "precision": p, "recall": r, "f1": f1,
             "fp_rate_technical": mean_over_runs(pnon[pnon.technical], lambda c: (c.mean(),))[0],
             "fp_rate_non_technical": mean_over_runs(pnon[~pnon.technical], lambda c: (c.mean(),))[0]}]
    for name, col in BEST.items():
        p, r, f1 = prf(pred.label, pred[col])
        rows.append({"model": name, "precision": p, "recall": r, "f1": f1,
                     "fp_rate_technical": round(non[non.technical][col].mean(), 3),
                     "fp_rate_non_technical": round(non[~non.technical][col].mean(), 3)})
    share = round(100 * non.technical.mean(), 1)
    return share, pd.DataFrame(rows)


def analysis_4(pred):
    sol = pred[pred.label == 1]; non = pred[pred.label == 0]
    s, r, l = (sol[BEST[k]] for k in ["SVM", "RoBERTa", "Llama-3_ft"])
    llama_fn = l == 0
    llama_fp = non[BEST["Llama-3_ft"]] == 1
    any_detects = (s + r + l) >= 1
    res = {
        "solution_comments": len(sol),
        "llama_missed": int(llama_fn.sum()),
        "llama_missed_also_missed_by_both_others": int((llama_fn & (s == 0) & (r == 0)).sum()),
        "llama_missed_caught_by_both_others": int((llama_fn & (s == 1) & (r == 1)).sum()),
        "llama_false_positives": int(llama_fp.sum()),
        "llama_fp_also_fp_of_at_least_one_other": int((llama_fp & ((non["SVC"] == 1) | (non["RoBERTa"] == 1))).sum()),
        "llama_fp_also_fp_of_both_others": int((llama_fp & (non["SVC"] == 1) & (non["RoBERTa"] == 1)).sum()),
        "llama_detected_but_both_others_missed": int(((l == 1) & (s == 0) & (r == 0)).sum()),
        "recall_if_any_model_detects": round(any_detects.mean(), 3),
    }
    return res


def analysis_5(pred, min_solution=20):
    rows = []
    for cat, g in pred.groupby("issue_category"):
        sol = g[g.label == 1]
        if len(sol) < min_solution:
            continue
        row = {"category": cat, "solution_comments": len(sol),
               "pct_code_descriptions": round(100 * (sol["code"] == "CODE_IMPLEMENTATION").mean(), 1)}
        for name, col in BEST.items():
            row["f1_" + name] = prf(g.label, g[col])[2]
        row["f1_mean"] = round(sum(row["f1_" + n] for n in BEST) / len(BEST), 3)
        rows.append(row)
    d = pd.DataFrame(rows).sort_values("f1_mean")
    rho, p = spearmanr(d.pct_code_descriptions, d.f1_mean)
    return d, round(rho, 2), round(p, 3)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    comments = pd.read_csv(COMMENT_DATA_PATH)
    pred = load_predictions()
    prompt = load_prompt_runs()

    a1a = analysis_1a(comments); a1a.to_csv(out("1a_distinctive_words.csv"), index=False)
    p1a = analysis_1a_phrase(comments); pd.Series(p1a).to_csv(out("1a_created_attachment_phrase.csv"), header=False)
    a1b = analysis_1b(pred, prompt); a1b.to_csv(out("1b_recall_per_solution_code.csv"), index=False)
    s2, e2 = analysis_2(comments); e2.to_csv(out("2_smote_effect_per_model.csv"))
    pd.Series(s2).to_csv(out("2_imbalance_summary.csv"), header=False)
    share3, a3 = analysis_3(pred, prompt); a3.to_csv(out("3_prompting_and_technical_fp.csv"), index=False)
    a4 = analysis_4(pred); pd.Series(a4).to_csv(out("4_error_overlap.csv"), header=False)
    a5, rho, p5 = analysis_5(pred); a5.to_csv(out("5_categories.csv"), index=False)

    pd.set_option("display.width", 200)
    print("test comments:", len(pred), "| solution:", int(pred.label.sum()))
    print("\n1a\n", a1a.to_string(index=False))
    print("\n1a (\"Created attachment\")\n", p1a)
    print("\n1b\n", a1b.to_string(index=False))
    print("\n2\n", s2, "\n", e2.to_string())
    print(f"\n3 (non-solution comments with technical content: {share3}%)\n", a3.to_string(index=False))
    print("\n4\n", a4)
    print(f"\n5 (Spearman rho={rho}, p={p5})\n", a5.to_string(index=False))


if __name__ == "__main__":
    main()
