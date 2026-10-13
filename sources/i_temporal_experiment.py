from utils import *

TEMPORAL_FOLD = 10

def group_data(input_data_path, temporal_folds_path):
    data = pd.read_csv(input_data_path)

    # Sort the data by date
    data = data.sort_values(by='creation_time')

    # Group the data into temporal folds
    fold_size = len(data) // TEMPORAL_FOLD
    for i in range(TEMPORAL_FOLD):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < (TEMPORAL_FOLD-1) else len(data)
        fold_data = data.iloc[start_idx:end_idx]

        #randomly shuffle the data in each fold
        fold_data = fold_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # make sure the temporal_folds_path directory exists
        os.makedirs(temporal_folds_path, exist_ok=True)

        fold_data.to_csv(f"{temporal_folds_path}/fold_{i}.csv", index=False)

def add_embedding_info(embedding_comment_data, mapped_comment_data, embedding_comment_data_temporal):
    embedding_data = pd.read_csv(embedding_comment_data)
    mapped_data = pd.read_csv(mapped_comment_data)

    # check if any of issue_id-text_id pair are duplicated in embedding_data or mapped_data
    if embedding_data.duplicated(subset=['issue_id', 'text_id']).any():
        raise ValueError("Duplicate issue_id-text_id pairs found in embedding_comment_data")
    if mapped_data.duplicated(subset=['issue_id', 'text_id']).any():
        raise ValueError("Duplicate issue_id-text_id pairs found in mapped_comment_data")

    for i, row in mapped_data.iterrows():
        issue_id = row['issue_id']
        text_id = row['text_id']
        embedding_row = embedding_data[(embedding_data['issue_id'] == issue_id) & (embedding_data['text_id'] == text_id)]
        if not embedding_row.empty:
            mapped_data.at[i, 'gpt_embedding'] = embedding_row.iloc[0]['gpt_embedding']
        else:
            raise ValueError(f"\n\nNo embedding found for issue_id {issue_id} and text_id {text_id}\n\n")
    
    print(f"Number of rows in mapped data: {len(mapped_data)}")
    print(f"Columns in mapped data: {mapped_data.columns.tolist()}")
    # print(f"top 20 rows of mapped data: {mapped_data.head(20)}")

    # Save the mapped data with embedding information
    mapped_data.to_csv(embedding_comment_data_temporal, index=False)

def train_test_split_data(input_path, output_path):
    for i in range(TEMPORAL_FOLD-1):
        train_data = pd.read_csv(f"{input_path}/fold_{i}.csv")

        for j in range(i):
            prev_fold_data = pd.read_csv(f"{input_path}/fold_{j}.csv")
            train_data = pd.concat([train_data, prev_fold_data], ignore_index=True)

            # check if any of issue_id-text_id pair are duplicated in train_data
            if train_data.duplicated(subset=['issue_id', 'text_id']).any():
                raise ValueError(f"Duplicate issue_id-text_id pairs found in train_data for fold {i} after adding fold {j}")


        test_data = pd.read_csv(f"{input_path}/fold_{i+1}.csv")

        # check if any of issue_id-text_id pair are duplicated in test_data
        if test_data.duplicated(subset=['issue_id', 'text_id']).any():
            raise ValueError(f"Duplicate issue_id-text_id pairs found in test_data for fold {i}")

        print(f" length of train data for fold {i}: {len(train_data)}; length of test data for fold {i}: {len(test_data)}")

        # Save the split data
        os.makedirs(f"{output_path}/{i}", exist_ok=True)

        train_data.to_csv(f"{output_path}/{i}/comment_train_data.csv", index=False)
        test_data.to_csv(f"{output_path}/{i}/comment_test_data.csv", index=False)

if __name__ == "__main__":
    add_embedding_info(embedding_comment_data, mapped_comment_data, embedding_comment_data_temporal)
    group_data(embedding_comment_data_temporal, temporal_folds_path)
    train_test_split_data(temporal_folds_path, temporal_folds_path)