import pandas as pd

# Load the CSV files
df_with_embeddings = pd.read_csv("gnucash_with_embeddings.csv")
df_labels = pd.read_csv("gnucash.csv")

# Drop the existing label column if it exists, then merge on (issue_id, text_id)
df_updated = df_with_embeddings.drop(columns=["label"], errors='ignore').merge(
    df_labels[["issue_id", "text_id", "label"]],
    on=["issue_id", "text_id"],
    how="left"
)

# Save the updated DataFrame
df_updated.to_csv("gnucash_with_updated_labels.csv", index=False)