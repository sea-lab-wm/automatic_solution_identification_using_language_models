import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np


# ---- Krippendorff’s Alpha (nominal, 2 annotators) ----
def krippendorffs_alpha(data):
    labels = list(set([x for row in data for x in row]))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(data)
    n_labels = len(labels)
    coincidence = np.zeros((n_labels, n_labels))
    for a1, a2 in data:
        i, j = label_to_idx[a1], label_to_idx[a2]
        coincidence[i, j] += 1
        coincidence[j, i] += 1  # symmetric
    Do = np.sum(coincidence * (1 - np.eye(n_labels))) / (2 * n)
    marginals = coincidence.sum(axis=0)
    De = np.sum(marginals[:, None] * marginals[None, :]) / (2 * n * n)
    return 1 - Do / De if De != 0 else 1.0


# ---- Main ----
def main(input_csv):
    df = pd.read_csv(input_csv)
    # Extract annotator labels
    a1 = df["Annotator-1"].tolist()
    a2 = df["Annotator-2"].tolist()
    # Compute metrics
    kappa = cohen_kappa_score(a1, a2)
    alpha = krippendorffs_alpha(list(zip(a1, a2)))
    agreement_rate = sum(x == y for x, y in zip(a1, a2)) / len(a1) * 100
    print(f"Agreement Rate: {agreement_rate:.2f}%")
    print(f"Cohen’s Kappa: {kappa:.3f}")
    print(f"Krippendorff’s Alpha: {alpha:.3f}")


if __name__ == "__main__":
    file_path = './qualitative_analysis.csv'
    main(file_path)