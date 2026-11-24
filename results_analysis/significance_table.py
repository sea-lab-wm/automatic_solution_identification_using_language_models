"""Builds the LaTeX table of the significance tests from generated/significance_tests/results.csv.

Each cell: difference A - B and its Holm-adjusted p-value; significant values (p < 0.05) in bold.
Paired cluster bootstrap over issues (issues as clusters) for all comparisons. The 95% CIs are not
reported in the paper, as they are not adjusted for multiple comparisons (pass ci=True to cell() to include them).

Usage: python results_analysis/significance_table.py
"""
import pandas as pd

IN = "results_analysis/generated/significance_tests/results.csv"
OUT = "results_analysis/generated/significance_tests/table.tex"
ROWS = [  # (claim_id, label in the paper)
    ("RQ1-emb-SVM", r"\svcs: \embllama vs. \embtfidf"),
    ("RQ1-emb-LR", r"\lrs: \embllama vs. \embtfidf"),
    ("RQ1-emb-KNN", r"\knns: \embllama vs. \embtfidf"),
    ("RQ1-emb-RF", r"\rfrs: \embllama vs. \embtfidf"),
    ("RQ1-emb-DT", r"\dtrees: \embllama vs. \embtfidf"),
    ("RQ1-emb-NB", r"\gnbs: \embllama vs. \embtfidf"),
    ("RQ1-svm-gpt", r"\svcs: \embgpt vs. \embtfidf"),
    ("RQ1-avg-tfidf", r"Average of the six models: \embllama vs. \embtfidf"),
    ("RQ1-avg-gpt", r"Average of the six models: \embllama vs. \embgpt"),
    ("RQ1-svm-lr", r"\svcs vs. \lrs"),
    ("RQ2-cot-zs", r"CoT vs. ZS"),
    ("RQ2-cot-fs", r"CoT vs. FS"),
    ("RQ2-desc-ZS", r"ZS: with vs. without issue description"),
    ("RQ2-desc-FS-7", r"FS: with vs. without issue description"),
    ("RQ2-desc-CoT-7", r"CoT: with vs. without issue description"),
    ("RQ2-prompt-svm", r"Prompted \llama (CoT) vs. \svcs"),
    ("RQ3-rob-bert", r"\rbert vs. \bert"),
    ("RQ3-rob-xlnet", r"\rbert vs. \xlnet"),
    ("RQ3-rob-distilbert", r"\rbert vs. \dbert"),
    ("RQ3-llama-rob", r"\llamaft vs. \rbert"),
    ("RQ3-llama-svm", r"\llamaft vs. \svcs"),
    ("RQ3-rob-svm", r"\rbert vs. \svcs"),
    ("RQ3-llama-prompt", r"\llamaft vs. prompted \llama"),
    ("RQ3-rob-prompt", r"\rbert vs. prompted \llama"),
    ("RQ4-ens-llama", r"Best ensemble vs. \llamaft"),
    ("TMP-rob-llama", r"\rbert vs. \llamaft"),
    ("TMP-roberta-svm", r"\rbert vs. \svcs"),
    ("TMP-llama-svm", r"\llamaft vs. \svcs"),
    ("TMP-drop-svm", r"\svcs: temporal vs. cross-validation"),
    ("TMP-drop-roberta", r"\rbert: temporal vs. cross-validation"),
    ("TMP-drop-llama", r"\llamaft: temporal vs. cross-validation"),
]
FAMILY_LABEL = {"RQ1": r"\ref{rq:embed}", "RQ2": r"\ref{rq:prompt}", "RQ3": r"\ref{rq:finetune}",
                "RQ4": r"\ref{rq:ensemble}", "Temporal": "Temp."}


def pval(p):
    # with B = 10,000 resamples, Holm-adjusted p-values cannot be resolved below about 0.001
    return "$\\leq$0.001" if p <= 0.001 + 1e-12 else f"{p:.3f}"


def cell(r, ci=False):
    test = "boot"
    p = r.boot_p_holm
    num = lambda v: f"{v:.3f}".replace("-", "$-$")
    txt = ("+" if r.delta >= 0 else "") + num(r.delta)
    if ci and test == "boot":
        txt += f" [{num(r.boot_ci_low)}, {num(r.boot_ci_high)}]"
    # Only the difference is shown; bold marks a significant difference (Holm-adjusted p < 0.05).
    # The exact p-values are in results.csv.
    return (r"\textbf{" + txt + "}") if p < 0.05 else txt


def main():
    res = pd.read_csv(IN)
    lines, last_family = [], None
    for cid, label in ROWS:
        g = res[res.claim_id == cid].set_index("metric")
        fam = g.family.iloc[0]
        if last_family is not None and fam != last_family:
            lines.append(r"\midrule")
        first = FAMILY_LABEL[fam] if fam != last_family else ""
        last_family = fam
        lines.append(f"{first} & {label} & {cell(g.loc['precision'])} & {cell(g.loc['recall'])} & "
                     f"{cell(g.loc['f1'])} \\\\")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
