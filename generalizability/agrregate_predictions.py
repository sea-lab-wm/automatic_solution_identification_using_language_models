import pandas as pd
import glob
import os


if __name__ == "__main__":
    projects = ['Gnucash', 'Chromium']
    for project in projects:
        # Path to the directory containing all the CSV files
        mlm_predictions_folder = f'generalizability/predictions/ml/SVC/{project}'
        plm_predictions_folder = f'generalizability/predictions/plm/roberta-base/{project}'

        output_file_path =f'generalizability/predictions/{project}-exps.csv'

        # Get a list of all CSV files in the directory
        mlm_prediction_files = glob.glob(os.path.join(mlm_predictions_folder, '*.csv'))
        plm_prediction_files = glob.glob(os.path.join(plm_predictions_folder, '*.csv'))

        all_prediction_files = mlm_prediction_files + plm_prediction_files

        # Sort files for consistency (optional)
        all_prediction_files.sort()

        # Read the first file as the base dataframe
        base_df = pd.read_csv(all_prediction_files[0])

        columns_to_drop = ['bert_embedding', 'gpt_embedding', 'llama_embedding']  # replace with your actual column names
        base_df = base_df.drop(columns=[col for col in columns_to_drop if col in base_df.columns])

        # Get the name of the prediction column from the file itself
        # first_model_name = base_df.columns[-1]

        # Keep the original name for the first model column
        # No renaming necessary if we're using actual column names

        # Loop through the remaining CSV files
        for file in all_prediction_files[1:]:
            df = pd.read_csv(file)

            # Extract the model's prediction column name from the last column
            model_name = df.columns[-1]

            # Merge with the base dataframe on (issue_id, text_id)
            base_df = base_df.merge(df[['issue_id', 'text_id', model_name]], on=['issue_id', 'text_id'], how='left')

        # Save the final combined dataframe
        base_df.to_csv(output_file_path, index=False)

        print(f'Aggregated predictions saved to: {output_file_path}')
