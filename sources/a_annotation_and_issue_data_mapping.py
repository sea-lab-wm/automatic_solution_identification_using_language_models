import re

from sklearn.model_selection import StratifiedKFold
import validators

from utils import *

def flatten_annotation_comment_data(annotation_data_path, flatten_annotation_comment):
    with open(annotation_data_path, 'r') as f:
        issues = json.load(f)

    result = [
        ['issue_id', 'text_id', 'text', 'code', 'label']
    ]

    for issue in issues:
        count = 1
        for i, annotation in enumerate(issue['annotations']):
            result.append([
                issue['issue_id'],
                count,
                annotation['quote'],
                annotation['code'],
                1 if annotation['code'] in solution_codes else 0
            ])

            count += 1


    with open(flatten_annotation_comment, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)

    print(f"File saved successfully to {flatten_annotation_comment}")

def flatten_issue_comment(input_path, output_path):
    with open(input_path, 'r') as f:
        issues = json.load(f)

    result = [
        ['issue_id', 'text_id', 'text', 'code', 'label']
    ]

    for issue in issues:
        count = 1

        for comment in issue['comments']:
            result.append([
                issue['id'],
                count,
                comment['text'],
                "UNCODED",
                -1
            ])
            count += 1

    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(result)

    print(f"File saved successfully to {output_path}")


def filterNonStringInstance(text):
    return isinstance(text, str) and len(text) > 0

def remove_reply(value):
    if not value or not isinstance(value, str):
        return value
    lines = value.split("\n")
    rem_reply_lines = [line for line in lines if
                       len(line) > 0 and line[0] != '>' and not line.startswith('(In reply to')]

    return '\n'.join(rem_reply_lines)


def read_issue_data(issue_data):
    issue_data = pd.read_csv(issue_data)

    print(f"Issue Data Length(Before Filter): {len(issue_data)}")

    mask = issue_data['text'].apply(filterNonStringInstance)

    filtered_out_data = issue_data[~mask]
    issue_data = issue_data[mask]

    print(f"Total empty issue data: {len(filtered_out_data)}")
    print(f"Empty data distribution: {filtered_out_data['label'].value_counts()}")

    issue_data['text'] = issue_data['text'].apply(lambda x: remove_reply(x))
    issue_data['matching_category'] = 'UNMATCHED'
    print(f"Issue Data Length(After Filter): {len(issue_data)}")
    return issue_data



def filterPushBy(text):
    text = text.strip("\"")
    text = text.strip("\'")

    if re.search(r"^Pushed by.*r=.*$", text):
        return False

    return True


def filterReviewRequestUpdated(text):
    text = text.strip()
    text = text.strip("\"")
    text = text.strip("\'")

    if re.search(r"^Review request updated.*$", text):
        return False

    return True


def filterReviewCommit(text):
    text = text.strip()
    text = text.strip("\"")
    text = text.strip("\'")

    if re.search(r"^Review commit.*$", text):
        return False

    return True


def filterAttachmentWithoutTextForAnnotation(text):
    text = text.strip("\"")
    text = text.strip("\'")
    if text.startswith('Attached') and (
            text.endswith("Details") or text.endswith("Splinter Review")):  # Remove Only automated text
        return False

    if re.search(r"^Attached.*Depends on D\d*$", text):
        return False

    return True


def filterNonStringInstance(text):
    return isinstance(text, str) and len(text) > 0


def filterLink(text):
    # Split the text by spaces
    components = text.split()
    # Flag to check if all components are URLs
    all_urls = True

    # Check each component
    for component in components:
        if not validators.url(component):
            return True  # As soon as we find a non-URL, return True
        all_urls = all_urls and validators.url(component)

    return not all_urls  # If all components are URLs, return False


def filterAttachmentWithoutHumanText(text):
    return not text.startswith('Created attachment') or not len(text.split('\n\n')) == 1


def read_annotation_data(annotation_data, filter):
    annotation_data = pd.read_csv(annotation_data)
    print(f"Annotation Data Length(Before Filter): {len(annotation_data)}")
    print(f"Annotation data label distribution: {annotation_data['label'].value_counts()}")

    if filter:
        annotation_data['code'] = annotation_data.apply(
            lambda row: "AUTO_LINK_" + row['code'] if not filterLink(row['text']) else row['code'],
            axis=1
        )
        annotation_data['code'] = annotation_data.apply(
            lambda row: "AUTO_ATTACH_" + row['code'] if not filterAttachmentWithoutTextForAnnotation(row['text']) else row[
                'code'],
            axis=1
        )
        annotation_data['code'] = annotation_data.apply(
            lambda row: "AUTO_PUSH_" + row['code'] if not filterPushBy(row['text']) else row['code'],
            axis=1
        )
        annotation_data['code'] = annotation_data.apply(
            lambda row: "AUTO_REV_COM_" + row['code'] if not filterReviewCommit(row['text']) else row['code'],
            axis=1
        )
        annotation_data['code'] = annotation_data.apply(
            lambda row: "AUTO_REV_REQ_UP_" + row['code'] if not filterReviewRequestUpdated(row['text']) else row['code'],
            axis=1
        )

        annotation_data['label'] = annotation_data.apply(lambda row: 0 if row['code'].startswith("AUTO_") else row['label'],
                                                         axis=1)

    print(f"Annotation Data Length(After Filter): {len(annotation_data)}")
    print(f"Annotation data label distribution")
    print(annotation_data['label'].value_counts())

    return annotation_data

def analyze_token_counts(texts):
    token_counts = [len(tokenize_without_punctuation(text)) for text in texts if
                    len(tokenize_without_punctuation(text)) != 0]

    return boxplot_stat(token_counts)

def boxplot_stat(token_counts):
    # Calculate summary statistics
    min_val = np.min(token_counts)
    max_val = np.max(token_counts)
    median_val = np.median(token_counts)
    q1_val = np.percentile(token_counts, 25)
    q3_val = np.percentile(token_counts, 75)

    # Calculate interquartile range (IQR)
    iqr = q3_val - q1_val

    # Calculate upper and lower bounds
    upper_bound = q3_val + 1.5 * iqr
    lower_bound = q1_val - 1.5 * iqr

    return max_val, median_val, min_val, q1_val, q3_val, upper_bound, lower_bound


def map_full_match(annotation_data, issue_data, match_type='text', matching_category=None):
    mismatched_annotations = []
    invalid_data_in_issue = []
    invalid_data_in_annotation = []

    multiple_annotation_match_count = 0
    multiple_matches = {}

    issue_data['is_matched'] = 0
    annotation_data['is_matched'] = 0

    for _, row_a in annotation_data.iterrows():
        multiple_matches[(row_a['issue_id'], row_a['text_id'], row_a['code'])] = []

    for issue_id in tqdm(annotation_data['issue_id'].unique(), desc="Mapping between issues and annotations"):
        selected_issues = issue_data[issue_data['issue_id'] == issue_id]

        selected_annotations = annotation_data[annotation_data['issue_id'] == issue_id]

        for index_a, row_a in selected_annotations.iterrows():

            annotation_tokens = tokenize_without_punctuation(row_a['text'])
            annotation_data_tokenized = ' '.join(annotation_tokens)

            if not isinstance(row_a['text'], str) or len(annotation_tokens) < 1:
                invalid_data_in_annotation.append(row_a)
                continue

            multiple_issue_match_count = 0
            for index, row in selected_issues.iterrows():

                issue_tokens = tokenize_without_punctuation(row['text'])
                issue_data_tokenized = ' '.join(issue_tokens)

                if not isinstance(row['text'], str) or len(issue_tokens) < 1:
                    invalid_data_in_issue.append(row)
                    continue

                if match_type == 'token':
                    if annotation_data_tokenized == issue_data_tokenized or annotation_data_tokenized in issue_data_tokenized:
                        multiple_issue_match_count += 1
                        multiple_matches[(row_a['issue_id'], row_a['text_id'], row_a['code'])].append(row)

                        selected_annotations.loc[index_a, 'is_matched'] = 1
                        annotation_data.loc[index_a, 'is_matched'] = 1

                        if issue_data.at[index, 'code'] in solution_codes and row_a['code'] not in solution_codes:
                            continue

                        issue_data.loc[index, 'code'] = row_a['code']
                        issue_data.loc[index, 'is_matched'] = 1
                        issue_data.loc[index, 'matching_category'] = matching_category
                        break
                elif match_type == 'text':
                    if row_a['text'] == row['text'] or row_a['text'] in row['text']:
                        multiple_issue_match_count += 1
                        multiple_matches[(row_a['issue_id'], row_a['text_id'], row_a['code'])].append(row)

                        selected_annotations.loc[index_a, 'is_matched'] = 1
                        annotation_data.loc[index_a, 'is_matched'] = 1

                        if issue_data.at[index, 'code'] in solution_codes and row_a['code'] not in solution_codes:
                            continue

                        issue_data.loc[index, 'code'] = row_a['code']
                        issue_data.loc[index, 'is_matched'] = 1
                        issue_data.loc[index, 'matching_category'] = matching_category
                        break

            if multiple_issue_match_count > 1:
                multiple_annotation_match_count += 1

        mismatched_data = selected_annotations[selected_annotations['is_matched'] == 0]
        for index_a, row_a in mismatched_data.iterrows():
            mismatched_annotations.append(
                [row_a['issue_id'], row_a['text_id'], row_a['text'], row_a['code'], row_a['label']])

    multiple_matches_sol = [{i: [j['text'] for j in multiple_matches[i]]} for i in multiple_matches.keys() if
                            len(multiple_matches[i]) > 1 and i[2] in solution_codes]
    multiple_matches_non_sol = [{i: [j['text'] for j in multiple_matches[i]]} for i in multiple_matches.keys() if
                                len(multiple_matches[i]) > 1 and i[2] not in solution_codes]
    print('-' * 20, matching_category, "-" * 20)
    print(f"\nTotal invalid annotations: {len(invalid_data_in_annotation)}")
    print(f"\nTotal invalid Issues: {len(invalid_data_in_issue)}")

    print(f"\nTotal Matched Annotations: {len(annotation_data[annotation_data['is_matched'] == 1])}")
    print(
        f"Total Matched Annotations (SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 1) & (annotation_data['label'] == 1)])}")
    print(
        f"Total Matched Annotations (NON-SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 1) & (annotation_data['label'] == 0)])}")

    print(f"\nTotal Mis-matched Annotations: {len(annotation_data[annotation_data['is_matched'] == 0])}")
    print(
        f"Total Mis-matched Annotations (SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 0) & (annotation_data['label'] == 1)])}")
    print(
        f"Total Mis-matched Annotations (NON-SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 0) & (annotation_data['label'] == 0)])}")

    print(f"\nTotal Multiple Annotation Match Count: {len(multiple_matches_sol) + len(multiple_matches_non_sol)}")
    print(f"Total Multiple Annotation Match Count (SOLUTION): {len(multiple_matches_sol)}")
    print(f"Total Multiple Annotation Match Count (NON-SOLUTION): {len(multiple_matches_non_sol)}")
    return issue_data, mismatched_annotations


def remove_prefix_postfix(text, prefix, postfix=''):
    # pattern = re.compile(rf'{prefix}(.*?){postfix}', re.DOTALL)
    # match = pattern.search(text)
    # return match.group(1).strip() if match else text

    pattern = re.compile(rf'{prefix}.*?{postfix}', re.DOTALL)
    return pattern.sub('', text).strip()


def remove_prefix(text, starting_substring):
    pattern = re.compile(re.escape(starting_substring) + r'(.*)')
    match = re.search(pattern, text)
    return match.group(1) if match else text


def preprocess_annotation(res):
    if not isinstance(res, str) or not res:
        return ""
    res = remove_prefix_postfix(res, 'Attached file', '\(obsolete\) — Details — Splinter Review')
    res = remove_prefix_postfix(res, 'Attached file', '\(obsolete\) — Details')
    res = remove_prefix_postfix(res, 'Attached file', '— Details')
    res = remove_prefix_postfix(res, 'Attached file', 'Details — Splinter Review')
    res = remove_prefix_postfix(res, 'Attached file', 'Splinter Review')

    res = remove_prefix_postfix(res, 'Attached patch', '\(obsolete\) — Details — Splinter Review')
    res = remove_prefix_postfix(res, 'Attached patch', 'Details — Splinter Review')
    res = remove_prefix_postfix(res, 'Attached patch', 'Splinter Review')

    res = res.replace('(obsolete) — Details — Splinter Review', ' ')
    res = res.replace('Details — Splinter Review', ' ')

    res = remove_prefix(res, 'Attached file')
    res = remove_prefix(res, 'Attached patch')
    return res


def map_partial_token_match(annotation_data, issue_data, word_count, accepted_match, matching_category=None,
                            preprocess=False):
    mismatched_annotations = []

    invalid_data_in_issue = []
    invalid_data_in_annotation = []

    multiple_annotation_match_count = 0
    issue_data['is_matched'] = 0
    annotation_data['is_matched'] = 0

    multiple_matches = {}
    for _, row_a in annotation_data.iterrows():
        multiple_matches[(row_a['issue_id'], row_a['text_id'], row_a['code'])] = []

    for issue_id in tqdm(annotation_data['issue_id'].unique(), desc="Mapping between issues and annotations"):
        selected_issues = issue_data[issue_data['issue_id'] == issue_id]

        selected_annotations = annotation_data[annotation_data['issue_id'] == issue_id]

        for index_a, row_a in selected_annotations.iterrows():

            annotation_tokens = tokenize_without_punctuation(row_a['text'])
            annotation_data_tokenized = ' '.join(annotation_tokens)

            if not isinstance(row_a['text'], str) or len(annotation_tokens) < 1:
                invalid_data_in_annotation.append(row_a)
                continue

            max_similarity = -float('inf')
            most_similar_issue = None

            multiple_issue_match_count = 0
            for index, row in selected_issues.iterrows():

                issue_tokens = tokenize_without_punctuation(row['text'])
                issue_data_tokenized = ' '.join(issue_tokens)

                if not isinstance(row['text'], str) or len(issue_tokens) < 1:
                    invalid_data_in_issue.append(row)
                    continue

                if preprocess:
                    preprocessed_text = preprocess_annotation(row_a['text'])
                    token_count = len(tokenize_without_punctuation(preprocessed_text))
                    similarity_score = calculate_token_similarity(preprocessed_text, row['text'])
                else:
                    token_count = len(tokenize_without_punctuation(row_a['text']))
                    similarity_score = calculate_token_similarity(row_a['text'], row['text'])

                if token_count >= word_count:
                    if similarity_score > accepted_match:
                        multiple_issue_match_count += 1
                        multiple_matches[(row_a['issue_id'], row_a['text_id'], row_a['code'])].append(row)

                        selected_annotations.loc[index_a, 'is_matched'] = 1
                        annotation_data.loc[index_a, 'is_matched'] = 1

                        if issue_data.at[index, 'code'] in solution_codes and row_a['code'] not in solution_codes:
                            continue

                        issue_data.loc[index, 'code'] = row_a['code']
                        issue_data.loc[index, 'is_matched'] = 1
                        issue_data.loc[index, 'matching_category'] = matching_category
                        break

                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        most_similar_issue = row

            if multiple_issue_match_count > 1:
                multiple_annotation_match_count += 1

        mismatched_data = selected_annotations[selected_annotations['is_matched'] == 0]
        for index_a, row_a in mismatched_data.iterrows():
            mismatched_annotations.append(
                [row_a['issue_id'], row_a['text_id'], row_a['text'], row_a['code'], row_a['label']])

    multiple_matches_sol = [{i: [j['text'] for j in multiple_matches[i]]} for i in multiple_matches.keys() if
                            len(multiple_matches[i]) > 1 and i[2] in solution_codes]
    multiple_matches_non_sol = [{i: [j['text'] for j in multiple_matches[i]]} for i in multiple_matches.keys() if
                                len(multiple_matches[i]) > 1 and i[2] not in solution_codes]
    print(f"\nTotal invalid annotations: {len(invalid_data_in_annotation)}")
    print(f"\nTotal invalid Issues: {len(invalid_data_in_issue)}")

    print(f"\nTotal Matched Annotations: {len(annotation_data[annotation_data['is_matched'] == 1])}")
    print(
        f"Total Matched Annotations (SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 1) & (annotation_data['label'] == 1)])}")
    print(
        f"Total Matched Annotations (NON-SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 1) & (annotation_data['label'] == 0)])}")

    print(f"\nTotal Mis-matched Annotations: {len(annotation_data[annotation_data['is_matched'] == 0])}")
    print(
        f"Total Mis-matched Annotations (SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 0) & (annotation_data['label'] == 1)])}")
    print(
        f"Total Mis-matched Annotations (NON-SOLUTION): {len(annotation_data[(annotation_data['is_matched'] == 0) & (annotation_data['label'] == 0)])}")

    print(f"\nTotal Multiple Annotation Match Count: {len(multiple_matches_sol) + len(multiple_matches_non_sol)}")
    print(f"Total Multiple Annotation Match Count (SOLUTION): {len(multiple_matches_sol)}")
    print(f"Total Multiple Annotation Match Count (NON-SOLUTION): {len(multiple_matches_non_sol)}")
    return issue_data, mismatched_annotations


def map_manual(annotation_data, issue_data, manual_mapping_path, matching_category):
    # Read the manual mapping CSV into a DataFrame
    manual_mapping_df = pd.read_csv(manual_mapping_path)

    # Initialize mismatched_annotations list
    mismatched_annotations = []

    # Iterate over each row in manual_mapping_df
    for _, row in manual_mapping_df.iterrows():
        issue_id = row['issue_id']
        issue_text_id = row['issue_text_id']
        annotation_text_id = row['annotation_text_id']

        # Find the corresponding issue and annotation based on IDs
        selected_issue = issue_data[(issue_data['issue_id'] == issue_id) & (issue_data['text_id'] == issue_text_id)]
        selected_annotation = annotation_data[
            (annotation_data['issue_id'] == issue_id) & (annotation_data['text_id'] == annotation_text_id)]

        if len(selected_issue) == 0:
            print(f"\nIssue with issue_id={issue_id} and text_id={issue_text_id} not found.")
        elif len(selected_annotation) == 0:
            print(f"\nAnnotation with issue_id={issue_id} and text_id={annotation_text_id} not found.")
        else:
            # Update issue_data with annotation details
            issue_data.loc[(issue_data['issue_id'] == issue_id) & (issue_data['text_id'] == issue_text_id), 'code'] = \
                selected_annotation['code'].values[0]
            issue_data.loc[
                (issue_data['issue_id'] == issue_id) & (issue_data['text_id'] == issue_text_id), 'is_matched'] = 1
            issue_data.loc[
                (issue_data['issue_id'] == issue_id) & (
                        issue_data['text_id'] == issue_text_id), 'matching_category'] = matching_category

    return issue_data


def map_issue_annotations_comments(issue_data, annotation_data, mapping_data, mismatched_annotations_path, manual_mapping_path=None, annotation_filter=True):
    issue_data = read_issue_data(issue_data)

    annotation_data = read_annotation_data(annotation_data, filter=annotation_filter)

    max_val, median_val, min_val, q1_val, q3_val, upper_bound, lower_bound = analyze_token_counts(annotation_data['text'].tolist())

    print(f"(Annotation Data) Max Value: {max_val}")
    print(f"(Annotation Data) Median Value: {median_val}")
    print(f"(Annotation Data) Min Value: {min_val}")
    print(f"(Annotation Data) Q1 (25th percentile): {q1_val}")
    print(f"(Annotation Data) Q3 (75th percentile): {q3_val}")
    print(f"(Annotation Data) Upper Bound: {upper_bound}")
    print(f"(Annotation Data) Lower Bound: {lower_bound}")

    columns_list = ['issue_id', 'text_id', 'text', 'code', 'label']

    # Exact Text Match
    issue_data, mismatched_annotations = map_full_match(annotation_data, issue_data, 'text', 'FULL_TEXT')
    # print(f"Mapped Comments Length: {len(issue_data)}")
    # print(f"Mismatched Annotation Length: {len(mismatched_annotations)}")

    mismatched_annotations_df = pd.DataFrame(mismatched_annotations, columns=columns_list)

    # Full Token Match
    issue_data, mismatched_annotations = map_full_match(mismatched_annotations_df, issue_data, 'token', "FULL_TOKEN")
    print(f"Mapped Comments Length: {len(issue_data)}")
    print(f"Mismatched Annotation Length: {len(mismatched_annotations)}")

    mismatched_annotations_df = pd.DataFrame(mismatched_annotations, columns=columns_list)

    # 90% Token Match
    accepted_match = 0.9
    issue_data, mismatched_annotations = map_partial_token_match(mismatched_annotations_df, issue_data, q1_val // 2,
                                                                 accepted_match, "90_PCT_MATCH", False)

    print(f"Mapped Comments Length: {len(issue_data)}")
    print(f"Mismatched Annotation Length: {len(mismatched_annotations)}")

    mismatched_annotations_df = pd.DataFrame(mismatched_annotations, columns=columns_list)

    # 80% Token Match
    accepted_match = 0.80
    issue_data, mismatched_annotations = map_partial_token_match(mismatched_annotations_df, issue_data, q1_val // 2,
                                                                 accepted_match, "80_PCT_MATCH", False)

    print(f"Mapped Comments Length: {len(issue_data)}")
    print(f"Mismatched Annotation Length: {len(mismatched_annotations)}")

    mismatched_annotations_df = pd.DataFrame(mismatched_annotations, columns=columns_list)

    # 90% Token Match after preprocess
    issue_data, mismatched_annotations = map_partial_token_match(mismatched_annotations_df, issue_data, 0,
                                                                 0.95, "95_PCT_MATCH_PROCESSED", True)
    print(f"Mapped Comments Length: {len(issue_data)}")
    print(f"Mismatched Annotation Length: {len(mismatched_annotations)}")

    mismatched_annotations_df = pd.DataFrame(mismatched_annotations, columns=columns_list)

    # Manual Mapping

    if manual_mapping_path is not None:
        issue_data = map_manual(mismatched_annotations_df, issue_data, manual_mapping_path,
                                                       "MANUAL")

    issue_data['label'] = issue_data['code'].apply(lambda x: 1 if x in solution_codes else 0)
    issue_data.drop('is_matched', axis=1, inplace=True)
    issue_data.drop_duplicates(subset=['text', 'label'], keep='first', inplace=True)

    # stat = issue_data['label'].value_counts()
    # with open(data_distribution_path, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Issue_Annotation_Data', stat.get(1, 0), stat.get(0, 0),
    #                      'Merged Data after issue and annotation are mapped including mismatched annotation'])

    print('=' * 30)
    print(f"Annotations Count: {annotation_data.shape[0]}")
    print(f"Annotation labels count: {annotation_data['label'].value_counts()}")
    print(f"Issue data Count: {issue_data.shape[0]}")
    print(f"Issue data labels count: {issue_data['label'].value_counts()}")
    print(f"Mapped data Count: {issue_data.shape[0]}")
    print(f"Mapped data labels Count: {issue_data['label'].value_counts()}")
    print(f"Mismatched Comment Count: {mismatched_annotations_df.shape[0]}")
    print(f"Mismatched Comments labels Count: {mismatched_annotations_df['label'].value_counts()}")
    print('=' * 30)

    issue_data.to_csv(mapping_data, index=False)
    mismatched_annotations_df.to_csv(mismatched_annotations_path, index=False)

    print("File saved successfully")
      
def create_comment_dataset():
    flatten_annotation_comment_data(annotation_data_path, flatten_annotation_comment)
    flatten_issue_comment(issue_data_path, flatten_issue_comment_data)
    map_issue_annotations_comments(flatten_issue_comment_data, flatten_annotation_comment, mapped_comment_data, mismatched_annotations_comments, manual_comment_mapping_path)
    save_mapped_data(data_split_comment_data, train_test_comment_data, 42)

def save_mapped_data(input_path, output_path, random_state=42):
    data = pd.read_csv(input_path)
    print(f"Data shape: {data.shape}")

    data = data[data['prompt_use'] == 0]

    solution_col = 'label'
    X = data.drop(solution_col, axis=1)
    y = data[solution_col]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    all_runs = {}

    for count, (train_index, test_index) in enumerate(cv.split(X, y)):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]

        run_data = {
            "train": data_train[['issue_id', 'text_id']].to_dict(orient='records'),
            "test": data_test[['issue_id', 'text_id']].to_dict(orient='records'),
        }

        all_runs[f"run{count + 1}"] = run_data

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(all_runs, json_file, indent=4)

if __name__ == '__main__':
    create_comment_dataset()