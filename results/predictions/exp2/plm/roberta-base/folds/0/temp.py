import pandas as pd

data = pd.read_csv("results/predictions/exp2/plm/roberta-base/folds/0/prediction_os_with_confidence.csv")
data = data[["issue_id","text_id","text","code","label","matching_category","roberta-base_os_prediction","roberta-base_os_confidence"]]
data.to_csv("results/predictions/exp2/plm/roberta-base/folds/0/prediction_os_with_confidence.csv", index=False)