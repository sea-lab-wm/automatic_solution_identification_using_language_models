from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd
import os
import math
import argparse

# Dictionary mapping prompt version to experiment names
exp_name_dict = {
    "1.1.0.3": "ZS_w/o_context",
    "1.1.1.3": "ZS_w_context",
    "1.2.0.3": "FS_4_exm_w/o_context",
    "1.2.0.4": "FS_7_exm_w/o_context",
    "1.2.1.3": "FS_4_exm_w_context",
    "1.2.1.4": "FS_7_exm_w_context",
    "1.3.0.3": "CoT_4xmpl_w/o_context",
    "1.3.0.4": "CoT_7xmpl_w/o_context",
    "1.3.1.3": "CoT_4xmpl_w_context",
    "1.3.1.4": "CoT_7xmpl_w_context"
}


# Function to compute evaluation metrics
def compute_metrics(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
    return cm, accuracy, precision, recall, f1, f2


# Function to average multiple sets of metrics
def average_metrics(metrics_list):
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = sum(d[key] for d in metrics_list) / len(metrics_list)
    return avg_metrics


# Function to calculate sample variance of multiple sets of metrics
def variance_metrics(metrics_list):
    var_metrics = {}
    n = len(metrics_list)
    if n > 1:  # Prevent division by zero if there's only one fold
        for key in metrics_list[0].keys():
            mean = sum(d[key] for d in metrics_list) / n
            var_metrics[key] = sum((d[key] - mean) ** 2 for d in metrics_list) / (n - 1)  # Sample variance
    else:
        # If there's only one fold, variance is undefined, return 0
        for key in metrics_list[0].keys():
            var_metrics[key] = 0
    return var_metrics


# Function to calculate sample standard deviation of multiple sets of metrics
def stddev_metrics(metrics_list):
    stddev_metrics = {}
    n = len(metrics_list)
    if n > 1:  # Prevent division by zero if there's only one fold
        for key in metrics_list[0].keys():
            mean = sum(d[key] for d in metrics_list) / n
            variance = sum((d[key] - mean) ** 2 for d in metrics_list) / (n - 1)  # Sample variance
            stddev_metrics[key] = math.sqrt(variance)  # Standard deviation is square root of variance
    else:
        # If there's only one fold, standard deviation is undefined, return 0
        for key in metrics_list[0].keys():
            stddev_metrics[key] = 0
    return stddev_metrics


def mean_metrics(metrics_list):
    mean_metrics = {}
    n = len(metrics_list)
    for key in metrics_list[0].keys():
        mean_metrics[key] = sum(d[key] for d in metrics_list) / n
    return mean_metrics


def minimum_metrics(metrics_list):
    min_metrics = {}
    for key in metrics_list[0].keys():
        min_metrics[key] = min(d[key] for d in metrics_list)
    return min_metrics


def maximum_metrics(metrics_list):
    max_metrics = {}
    for key in metrics_list[0].keys():
        max_metrics[key] = max(d[key] for d in metrics_list)
    return max_metrics


def max_min_diff_metrics(metrics_list):
    max_min_diff_metrics = {}
    for key in metrics_list[0].keys():
        max_val = max(d[key] for d in metrics_list)
        min_val = min(d[key] for d in metrics_list)
        max_min_diff_metrics[key] = max_val - min_val
    return max_min_diff_metrics


# Function to calculate metrics
def calculate_aggregated_metrics(df_group):
    TP = df_group['TP'].sum()
    FP = df_group['FP'].sum()
    TN = df_group['TN'].sum()
    FN = df_group['FN'].sum()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * (precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Aggregated',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'F2': f2,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    })


def calculate_mean(df_group):
    accuracy_mean = df_group['Accuracy'].mean()
    precision_mean = df_group['Precision'].mean()
    recall_mean = df_group['Recall'].mean()
    f1_mean = df_group['F1'].mean()
    f2_mean = df_group['F2'].mean()
    TP_mean = df_group['TP'].mean()
    FP_mean = df_group['FP'].mean()
    TN_mean = df_group['TN'].mean()
    FN_mean = df_group['FN'].mean()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Mean',
        'Accuracy': accuracy_mean,
        'Precision': precision_mean,
        'Recall': recall_mean,
        'F1': f1_mean,
        'F2': f2_mean,
        'TP': TP_mean,
        'FP': FP_mean,
        'TN': TN_mean,
        'FN': FN_mean
    })


# Function to calculate variance of metrics
def calculate_variance(df_group):
    accuracy_var = df_group['Accuracy'].var()
    precision_var = df_group['Precision'].var()
    recall_var = df_group['Recall'].var()
    f1_var = df_group['F1'].var()
    f2_var = df_group['F2'].var()
    TP_var = df_group['TP'].var()
    FP_var = df_group['FP'].var()
    TN_var = df_group['TN'].var()
    FN_var = df_group['FN'].var()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Variance',
        'Accuracy': accuracy_var,
        'Precision': precision_var,
        'Recall': recall_var,
        'F1': f1_var,
        'F2': f2_var,
        'TP': TP_var,
        'FP': FP_var,
        'TN': TN_var,
        'FN': FN_var
    })


def calculate_standard_deviation(df_group):
    accuracy_std = df_group['Accuracy'].std()
    precision_std = df_group['Precision'].std()
    recall_std = df_group['Recall'].std()
    f1_std = df_group['F1'].std()
    f2_std = df_group['F2'].std()
    TP_std = df_group['TP'].std()
    FP_std = df_group['FP'].std()
    TN_std = df_group['TN'].std()
    FN_std = df_group['FN'].std()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Standard Deviation',
        'Accuracy': accuracy_std,
        'Precision': precision_std,
        'Recall': recall_std,
        'F1': f1_std,
        'F2': f2_std,
        'TP': TP_std,
        'FP': FP_std,
        'TN': TN_std,
        'FN': FN_std
    })


def calculate_maximum(df_group):
    accuracy_max = df_group['Accuracy'].max()
    precision_max = df_group['Precision'].max()
    recall_max = df_group['Recall'].max()
    f1_max = df_group['F1'].max()
    f2_max = df_group['F2'].max()
    TP_max = df_group['TP'].max()
    FP_max = df_group['FP'].max()
    TN_max = df_group['TN'].max()
    FN_max = df_group['FN'].max()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Maximum',
        'Accuracy': accuracy_max,
        'Precision': precision_max,
        'Recall': recall_max,
        'F1': f1_max,
        'F2': f2_max,
        'TP': TP_max,
        'FP': FP_max,
        'TN': TN_max,
        'FN': FN_max
    })


def calculate_minimum(df_group):
    accuracy_min = df_group['Accuracy'].min()
    precision_min = df_group['Precision'].min()
    recall_min = df_group['Recall'].min()
    f1_min = df_group['F1'].min()
    f2_min = df_group['F2'].min()
    TP_min = df_group['TP'].min()
    FP_min = df_group['FP'].min()
    TN_min = df_group['TN'].min()
    FN_min = df_group['FN'].min()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Minimum',
        'Accuracy': accuracy_min,
        'Precision': precision_min,
        'Recall': recall_min,
        'F1': f1_min,
        'F2': f2_min,
        'TP': TP_min,
        'FP': FP_min,
        'TN': TN_min,
        'FN': FN_min
    })


def calculate_max_min_diff(df_group):
    accuracy_max = df_group['Accuracy'].max()
    accuracy_min = df_group['Accuracy'].min()
    precision_max = df_group['Precision'].max()
    precision_min = df_group['Precision'].min()
    recall_max = df_group['Recall'].max()
    recall_min = df_group['Recall'].min()
    f1_max = df_group['F1'].max()
    f1_min = df_group['F1'].min()
    f2_max = df_group['F2'].max()
    f2_min = df_group['F2'].min()
    TP_max = df_group['TP'].max()
    TP_min = df_group['TP'].min()
    FP_max = df_group['FP'].max()
    FP_min = df_group['FP'].min()
    TN_max = df_group['TN'].max()
    TN_min = df_group['TN'].min()
    FN_max = df_group['FN'].max()
    FN_min = df_group['FN'].min()

    return pd.Series({
        'Experiment Name': df_group['Experiment Name'].iloc[0],
        'Fold': 'Max-Min Diff',
        'Accuracy': accuracy_max - accuracy_min,
        'Precision': precision_max - precision_min,
        'Recall': recall_max - recall_min,
        'F1': f1_max - f1_min,
        'F2': f2_max - f2_min,
        'TP': TP_max - TP_min,
        'FP': FP_max - FP_min,
        'TN': TN_max - TN_min,
        'FN': FN_max - FN_min
    })


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate prompts and responses')
    parser.add_argument('dataset_type', type=str, help='Dataset types: development/test')
    args = parser.parse_args()

    # Set file paths based on dataset type
    if args.dataset_type == "development":
        data_split_path = '../dataset/solution_identification_data/train_test_split_prompt_set.json'
        response_folder_path = '../prompting/generated_responses_dev_set'
        result_file_path = '../results/llm_prompting/dev_set_results_all_folds.csv'
        aggregated_result_file_path = '../results/llm_prompting/dev_set_results_aggregated.csv'
    elif args.dataset_type == "test":
        data_split_path = '../dataset/solution_identification_data/train_test_split.json'
        response_folder_path = '../prompting/generated_responses_test_set_original'
        result_file_path = '../results/llm_prompting/test_set_results_all_folds.csv'
        aggregated_result_file_path = '../results/llm_prompting/test_set_results_aggregated.csv'

    prompt_versions = ["1.1.0.3", "1.1.1.3", "1.2.0.3", "1.2.0.4", "1.2.1.3", "1.2.1.4", "1.3.0.3", "1.3.1.3",
                       "1.3.0.4", "1.3.1.4"]
    results = []
    for prompt_version in prompt_versions:
        print(f"Processing responses for prompt version {prompt_version}")
        fold_folders = os.listdir(response_folder_path)
        fold_folders.sort()
        for fold_folder in fold_folders:
            if not fold_folder.startswith('fold'):
                continue
            fold = int(fold_folder.split('fold')[-1])
            response_file_path = os.path.join(response_folder_path, fold_folder, f"responses-{prompt_version}.csv")
            response_df = pd.read_csv(response_file_path)

            # Convert labels to a list of integers
            labels = response_df['label']
            labels = [int(label) for label in labels.tolist()]

            # Define the prediction columns for each model
            models = {
                "LLAMA": ['llama_prediction_0', 'llama_prediction_1', 'llama_prediction_2'],
            }

            # Process predictions for each model
            for model_name, prediction_columns in models.items():
                if all(col in response_df.columns for col in prediction_columns):
                    model_metrics = []
                    for col in prediction_columns:
                        predictions = response_df[col]
                        run = col.split('_')[-1]
                        # Compute metrics for each set of predictions
                        cm, accuracy, precision, recall, f1, f2 = compute_metrics(labels, predictions)
                        model_metrics.append({
                            "TP": cm[1, 1],
                            "FP": cm[0, 1],
                            "TN": cm[0, 0],
                            "FN": cm[1, 0],
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "F2": f2
                        })
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": run,
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            "TP": cm[1, 1],
                            "FP": cm[0, 1],
                            "TN": cm[0, 0],
                            "FN": cm[1, 0],
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1": f1,
                            "F2": f2
                        })

                    # Check if model_metrics is not empty before averaging and calculating variance
                    if model_metrics:
                        avg_metrics = average_metrics(model_metrics)
                        std_metrics = stddev_metrics(model_metrics)
                        var_metrics = variance_metrics(model_metrics)
                        # Store the averaged metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "All",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **avg_metrics
                        })
                        # Store the standard deviation metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Standard Deviation",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **std_metrics
                        })
                        # Store the variance metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Variance",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **var_metrics
                        })
                        # Store the mean metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Mean",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **mean_metrics(model_metrics)
                        })
                        # Store the maximum metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Maximum",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **maximum_metrics(model_metrics)
                        })
                        # Store the minimum metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Minimum",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **minimum_metrics(model_metrics)
                        })
                        # Store the maximum-minimum difference metrics
                        results.append({
                            "Model Name": model_name,
                            "Fold": fold,
                            "Run": "Max-Min Diff",
                            "Prompt Version": prompt_version,
                            "Experiment Name": exp_name_dict[prompt_version],
                            **max_min_diff_metrics(model_metrics)
                        })
                    else:
                        print(f"No metrics computed for model {model_name} with prompt version {prompt_version}")

    # Convert results to DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(result_file_path, index=False, float_format='%.4f')
    else:
        print("No results to save.")

    # After processing all folds, aggregate across all folds for each prompt version
    aggregated_results = []
    filtered_results_df = results_df[results_df['Run'] == 'All']
    # print(len(filtered_results_df))
    grouped_results_df = filtered_results_df.groupby(['Model Name', 'Prompt Version'])

    # Group by model_name and oversample, then calculate the aggregated metrics
    aggregated_results = grouped_results_df.apply(calculate_aggregated_metrics).reset_index()

    # Calculate the mean of metrics across the folds
    mean_results = grouped_results_df.apply(calculate_mean).reset_index()

    # Calculate the standard deviation of metrics across the folds
    std_results = grouped_results_df.apply(calculate_standard_deviation).reset_index()

    # Calculate the variance of metrics across the folds
    variance_results = grouped_results_df.apply(calculate_variance).reset_index()

    # Calculate the maximum of metrics across the folds
    max_results = grouped_results_df.apply(calculate_maximum).reset_index()

    # Calculate the minimum of metrics across the folds
    min_results = grouped_results_df.apply(calculate_minimum).reset_index()

    # Calculate the maximum-minimum difference of metrics across the folds
    max_min_diff_results = grouped_results_df.apply(calculate_max_min_diff).reset_index()

    # Combine the aggregated results and the variance results
    final_results = pd.concat([aggregated_results, mean_results, std_results, variance_results, max_results, min_results, max_min_diff_results])

    aggregated_results_df = pd.DataFrame(final_results)
    aggregated_results_df.to_csv(aggregated_result_file_path, index=False, float_format='%.4f')
