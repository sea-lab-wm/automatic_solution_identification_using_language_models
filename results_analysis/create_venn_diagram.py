import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3_unweighted

# Step 1: Load the data from CSV
result_path = '../results/three_best_models.csv'
df = pd.read_csv(result_path)  # Use sep="," if it's comma-separated

# Step 2: Identify False Positives (label=0, prediction=1)
fp_llama = set(df[(df['label'] == 0) & (df['LLAMA-FT'] == 1)].index)
fp_roberta = set(df[(df['label'] == 0) & (df['RoBERTa'] == 1)].index)
fp_svc = set(df[(df['label'] == 0) & (df['SVC'] == 1)].index)

# Step 3: Identify False Negatives (label=1, prediction=0)
fn_llama = set(df[(df['label'] == 1) & (df['LLAMA-FT'] == 0)].index)
fn_roberta = set(df[(df['label'] == 1) & (df['RoBERTa'] == 0)].index)
fn_svc = set(df[(df['label'] == 1) & (df['SVC'] == 0)].index)

# Step 4: Plot False Positives Venn Diagram
plt.figure(figsize=(8, 8))
venn_fp = venn3_unweighted([fp_llama, fp_roberta, fp_svc],
                           set_labels=(
                               f"LLAMA-FT\n({len(fp_llama)})",
                               f"RoBERTa\n({len(fp_roberta)})",
                               f"SVC\n({len(fp_svc)})"
                           ))
for text in venn_fp.subset_labels:
    if text: text.set_fontsize(18)
for text in venn_fp.set_labels:
    if text: text.set_fontsize(18)
plt.title("False Positives Venn Diagram\n", fontsize=22)
plt.tight_layout()
plt.show()

# Step 5: Plot False Negatives Venn Diagram
plt.figure(figsize=(8, 8))
venn_fn = venn3_unweighted([fn_llama, fn_roberta, fn_svc],
                           set_labels=(
                               f"LLAMA-FT\n({len(fn_llama)})",
                               f"RoBERTa\n({len(fn_roberta)})",
                               f"SVC\n({len(fn_svc)})"
                           ))
for text in venn_fn.subset_labels:
    if text: text.set_fontsize(18)
for text in venn_fn.set_labels:
    if text: text.set_fontsize(18)
plt.title("False Negatives Venn Diagram\n", fontsize=22)
plt.tight_layout()
plt.show()