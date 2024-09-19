import pandas as pd
import json
import os
from langchain_community.llms import Ollama
from tqdm import tqdm
from transformers import BertTokenizer
import argparse
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def generate_prompts(issue_data_path, dataset_path, data_split_path, prompt_versions, prompt_templates_path,
                     prompt_folder_path):
    """
    Generate prompts for all the 10 prompt versions using the prompt templates and the issue data.
    :param issue_data_path:
    :param dataset_path:
    :param data_split_path:
    :param prompt_versions:
    :param prompt_templates_path:
    :param prompt_folder_path:
    :return:
    """
    issue_data = json.load(open(issue_data_path))
    templates_df = pd.read_csv(prompt_templates_path)
    dataset_df = pd.read_csv(dataset_path)
    data_split_dict = json.load(open(data_split_path))

    fold = "run"
    for i in range(2, 11):
        fold += str(i)

        train_test_data = data_split_dict[fold]
        test_data = train_test_data["test"]

        fold_folder_path = os.path.join(prompt_folder_path, fold)
        if not os.path.exists(fold_folder_path):
            os.makedirs(fold_folder_path)
        for prompt_version in prompt_versions:
            generated_prompts_file_path = os.path.join(fold_folder_path, f"prompts-{prompt_version}.csv")
            # Iterate over all the prompt templates and get the desired template
            for _, template_row in templates_df.iterrows():
                if template_row['Version'] == prompt_version:
                    prompt_template = template_row['Template']
                    break

            n_tokens_in_prompt_template = tokenizer(prompt_template, return_tensors='pt')['input_ids'].shape[1]

            # Prepare a list to hold all the prompts
            prompt_rows = []

            # Iterate over all the sentences of the test data and generate the prompts
            for index, row in dataset_df.iterrows():
                issue_id = row['issue_id']
                text_id = row['text_id']

                is_test_data = 0
                for issue in test_data:
                    if issue['issue_id'] == issue_id and issue['text_id'] == text_id:
                        is_test_data = 1
                        break

                if is_test_data == 0:
                    continue

                # text = row['text']
                text = row['text']
                code = row['code']
                label = row['label']
                matching_category = row['matching_category']
                n_tokens_in_comment = tokenizer(text, return_tensors='pt')['input_ids'].shape[1]
                available_space = 2000 - n_tokens_in_prompt_template
                if n_tokens_in_comment > available_space:
                    text = text[:available_space]
                # Replace the placeholders in the template with the actual values
                prompt = prompt_template.replace('<comment_of_an_issue_report>', text)
                if prompt_version.split(".")[2] == "1":
                    for issue in issue_data:
                        if issue["id"] != issue_id:
                            continue
                        issue_title = issue["summary"]
                        for comment in issue["comments"]:
                            if comment["count"] == 0:
                                issue_description = comment["text"]
                                break
                        prompt = prompt.replace('<issue_title>', issue_title)
                        prompt = prompt.replace('<issue_description>', issue_description)
                        break

                prompt_rows.append([issue_id, text_id, text, code, label, matching_category, prompt])

            prompts_df = pd.DataFrame(prompt_rows,
                                      columns=['issue_id', 'text_id', 'text', 'code', 'label', 'matching_category',
                                               'prompt'])
            prompts_df.to_csv(generated_prompts_file_path, index=False)
        fold = "run"


def generate_response_with_llama(prompts_df, response_file_path, i):
    """
    Generate responses for the prompts using the LLAMA-3 model.
    :param prompts_df:
    :param response_file_path:
    :param i:
    :return:
    """
    llm = Ollama(model="llama3:70b")

    responses = []
    for index, row in tqdm(prompts_df.iterrows(), total=prompts_df.shape[0], desc='Getting LLAMA Predictions'):
        prompt = row['prompt']
        response = llm.invoke(prompt)
        # print(response)
        responses.append(response)

    prompts_df[f"llama_prediction_{i}"] = responses
    prompts_df.to_csv(response_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate prompts and responses')
    parser.add_argument('dataset_type', type=str, help='Dataset types: development/test')
    args = parser.parse_args()

    # Paths to the CSV file containing the prompt templates
    prompt_templates_path = './prompt_templates.csv'
    issue_data_path = '../../../dataset/issue_data/coded_issue_data.json'
    dataset_path = '../../dataset_construction/comment_data/actual_comment_data.csv'
    # dataset_path = '../../dataset_construction/comment_data/masked_comment_data_test_set.csv'

    if args.dataset_type == "development":
        data_split_path = '../../dataset_construction/comment_data/train_test_example_comment_data.json'
        prompt_folder_path = './generated_prompts_dev_set'
        response_folder_path = './generated_responses_dev_set'
    elif args.dataset_type == "test":
        data_split_path = '../../dataset_construction/comment_data/train_test_comment_data.json'
        prompt_folder_path = './generated_prompts'
        response_folder_path = './generated_responses'

    if not os.path.exists(prompt_folder_path):
        os.makedirs(prompt_folder_path)
    if not os.path.exists(response_folder_path):
        os.makedirs(response_folder_path)

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    prompt_versions = ["1.1.0.3", "1.1.1.3", "1.2.0.3", "1.2.0.4", "1.2.1.3", "1.2.1.4", "1.3.0.3", "1.3.1.3",
                       "1.3.0.4", "1.3.1.4"]

    # Generate prompts from the dataset
    generate_prompts(issue_data_path, dataset_path, data_split_path, prompt_versions, prompt_templates_path,
                     prompt_folder_path)

    # # Generate responses form prompt files
    for fold in os.listdir(prompt_folder_path):
        print(f"Processing {fold}")
        print("=====================================")
        if not fold.startswith('run'):
            continue
        fold_folder_path = os.path.join(prompt_folder_path, fold)
        for prompt_file in os.listdir(fold_folder_path):
            if not prompt_file.endswith('.csv'):
                continue
            prompt_version = prompt_file.split('-')[1].replace('.csv', '')
            if prompt_version not in prompt_versions:
                continue
            prompt_file_path = os.path.join(fold_folder_path, prompt_file)
            fold_response_folder_path = os.path.join(response_folder_path, fold)
            if not os.path.exists(fold_response_folder_path):
                os.makedirs(fold_response_folder_path)
            response_file_path = os.path.join(fold_response_folder_path, f"responses-{prompt_version}.csv")
            prompts_df = pd.read_csv(prompt_file_path)

            print(f"Generating responses for {prompt_file}")
            for i in range(3):
                generate_response_with_llama(prompts_df, response_file_path, i, device)
            print("\n")
