import csv
from tqdm import tqdm
import json
import os
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
import tensorflow as tf
import random
import torch
import en_core_web_lg
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool

nlp = en_core_web_lg.load()


def calculate_cosine_similarity(sentence1, sentence2):
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    cosine_sim = cosine_similarity(vectors)[0][1]
    return cosine_sim


def float_format(value):
    value *= 100
    value = round(value, 6)
    return value


def calculate_results(y_pred, y_test):
    accuracy = float_format(accuracy_score(y_test, y_pred))
    precision = float_format(precision_score(y_test, y_pred))
    recall = float_format(recall_score(y_test, y_pred))
    f1 = float_format(f1_score(y_test, y_pred))
    c = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = c.ravel()
    TN = int(TN)
    FP = int(FP)
    FN = int(FN)
    TP = int(TP)
    return FN, FP, TN, TP, accuracy, f1, precision, recall


def handle_null_values(df, df_name):
    if df.isnull().values.any():
        print(f"{df_name} contains null values")
        null_rows = df[df.isnull().any(axis=1)]
        print(f"Rows with null values in {df_name}:\n", null_rows)
        df.dropna(axis=0, inplace=True)


def get_train_test_data(data, indexes, run):
    run_key = f"run{run + 1}"
    if run_key not in indexes:
        raise ValueError("Invalid run number or data type")
    # Extract subsets
    train_subset = indexes[run_key]['train']
    test_subset = indexes[run_key]['test']
    # Create dataframes and merge
    train_df = pd.merge(pd.DataFrame(train_subset), data, on=['issue_id', 'text_id'],
                        how='left')
    test_df = pd.merge(pd.DataFrame(test_subset), data, on=['issue_id', 'text_id'],
                       how='left')
    return test_df, train_df


def append_row_to_csv(file_path, row):
    # Check if the file exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    
    # Open the file in append mode if it exists, otherwise open it in write mode to create it
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


def create_empty_csv(file_path, headers):

    # Check if the file exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"Empty CSV with headers {headers} created at {file_path}")


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def balance_classes(train_df, label_col, output_type='xy'):
    class_counts = train_df[label_col].value_counts()
    print("Class counts in the training set:")
    print(class_counts)

    # Identify the minority and majority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Get the minority class data
    minority_class_data = train_df[train_df[label_col] == minority_class]
    majority_class_data = train_df[train_df[label_col] == majority_class]

    # Calculate the number of duplicates needed to balance the classes
    num_duplicates = class_counts[majority_class] // class_counts[minority_class]
    remainder = class_counts[majority_class] % class_counts[minority_class]

    # Duplicate the minority class data
    duplicated_minority_class_data = pd.concat(
        [minority_class_data] * num_duplicates + [minority_class_data.head(remainder)])

    # Append the duplicated data to the original training set
    balanced_train_df = pd.concat([majority_class_data, duplicated_minority_class_data])

    # Shuffle the balanced training set
    balanced_train_df = balanced_train_df.reset_index(drop=True)

    print("Balanced class counts:")
    print(balanced_train_df[label_col].value_counts())

    if output_type == 'df':
        return pd.concat([balanced_train_df.drop(label_col, axis=1), balanced_train_df[label_col]], axis=1)
    else:
        return balanced_train_df.drop(label_col, axis=1), balanced_train_df[label_col]



def calculate_token_similarity(sentence1, sentence2):
    tokens1 = tokenize_without_punctuation(sentence1)
    tokens2 = tokenize_without_punctuation(sentence2)

    index2 = 0
    match_count = 0

    for token in tokens1:
        try:
            index2 = tokens2.index(token, index2)
            match_count += 1
            index2 += 1
        except ValueError:
            continue

    return match_count / len(tokens1) if tokens1 else 0


def set_seed(seed):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    
    print(f"Random seed set as {seed}")


def set_llm_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    
    print(f"Random seed set as {seed}")



def tokenize_without_punctuation(text):
    if not isinstance(text, str) or not text:
        return []
    tokens = word_tokenize(text)
    tokens = ["".join(char for char in token if char.isalnum()) for token in tokens]
    tokens = [token for token in tokens if token]

    return tokens


def spacy_parser(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_strings = [sentence.text for sentence in sentences]
    return sentence_strings


def get_experiment_config():
    with open(experiment_file, 'r') as file:
        experiments = json.load(file)
    return experiments


def clean_dataframe(data_path):
    data = pd.read_csv(data_path)
    required_columns = ['issue_id', 'text_id', 'text', 'code', 'label']
    optional_columns = ['bert_embedding', 'gpt_embedding', 'llama_embedding']
    columns_to_keep = required_columns + [col for col in optional_columns if col in data.columns]
    data = data[columns_to_keep]
    
    return data

def process_run(run, data, indexes, paths):
    dev_data_path, test_data_path, train_data_path, val_data_path, train_os_data_path, dev_os_df_path = paths

    test_df, dev_df = get_train_test_data(data, indexes, run)

    X_dev, y_dev = dev_df.drop('label', axis=1), dev_df['label']
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1, stratify=y_dev, shuffle=True, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    train_df_os = balance_classes(train_df, 'label', output_type='df')
    dev_df_os = balance_classes(dev_df, 'label', output_type='df')
    
    os.makedirs(os.path.dirname(dev_data_path.replace('<run>', f'{run}')), exist_ok=True)

    dev_df.to_csv(dev_data_path.replace('<run>', f'{run}'), index=False)
    test_df.to_csv(test_data_path.replace('<run>', f'{run}'), index=False)
    train_df.to_csv(train_data_path.replace('<run>', f'{run}'), index=False)
    val_df.to_csv(val_data_path.replace('<run>', f'{run}'), index=False)
    train_df_os.to_csv(train_os_data_path.replace('<run>', f'{run}'), index=False)
    dev_df_os.to_csv(dev_os_df_path.replace('<run>', f'{run}'), index=False)

    print(f"\n\nAll datas are saved successfully to {os.path.dirname(dev_data_path.replace('<run>', f'{run}'))}.")

def prepare_data(data_path, index_path):
    data = clean_dataframe(data_path)

    print(data['label'].value_counts())

    with open(index_path, 'r') as f:
        indexes = json.load(f)

    paths = [
        dev_data_path,
        test_data_path,
        train_data_path,
        val_data_path,
        train_os_data_path,
        dev_os_df_path
    ]

    # Create a pool of workers
    with Pool() as pool:
        pool.starmap(process_run, [(run, data, indexes, paths) for run in range(10)])



solution_codes = ["POTENTIAL_SOLUTION_DESIGN", "SOLUTION_REVIEW", "CODE_IMPLEMENTATION"]
data_setting = ["all", "filtered"]

experiment_file = 'experiments.json'

issue_data_path = "dataset_construction/comment_data/coded_issue_data.json"
issue_statistics = "dataset_construction/comment_data/issue_statistics.csv"
issue_comment_sentence = "dataset_construction/comment_data/issue_comment_sentence.json"

annotation_data_path = "dataset_construction/comment_data/annotation_data_all_codes.json"
annotation_statistics = "dataset_construction/comment_data/annotation_statistics.csv"
issue_annotation_sentence = "dataset/annotation_data/issue_annotation_sentence.json"

data_distribution_path = 'dataset_construction/comment_data/data_distribution.csv'

parsed_issue_data = 'dataset_construction/sentence_data/parsed_issue_data.csv'
flatten_issue_data = 'dataset_construction/comment_data/flatten_issue_data.csv'
flatten_issue_comment_data = 'dataset_construction/comment_data/flatten_issue_comment_data.csv'

parsed_annotation_data = 'dataset_construction/sentence_data/parsed_annotation_data.csv'
flatten_annotation_data = 'dataset_construction/comment_data/flatten_annotation_data.csv'
flatten_annotation_comment = 'dataset_construction/comment_data/flatten_annotation_comment_data.csv'
annotation_matching_data = "dataset_construction/comment_data/annotation_matching_data.csv"

mapped_sentence_data = 'dataset_construction/sentence_data/actual_sentence_data.csv'
manual_sentence_mapping_path = "dataset_construction/sentence_data/manual_sentence_mapping.csv"
mismatched_annotations_path = 'dataset_construction/sentence_data/mismatched_annotations_sentences.csv'

mapped_comment_data = 'dataset_construction/comment_data/actual_comment_data.csv'
data_split_comment_data = "dataset_construction/comment_data/example_actual_comment_data.csv"
embedding_comment_data = 'dataset_construction/comment_data/actual_comment_data_with_embeddings.csv'
promt_comment_data = 'dataset_construction/comment_data/comment_data.csv'
preprocess_comment_data = 'dataset_construction/comment_data/preprocess_comment_data.csv'
preprocess_sentence_data = 'dataset_construction/sentence_data/preprocess_sentence_data.csv'
manual_comment_mapping_path = "dataset_construction/comment_data/manual_comment_mapping.csv"
mismatched_annotations_comments = 'dataset_construction/comment_data/mismatched_annotations_comments.csv'

dataset_path = 'dataset'
dev_data_path = 'dataset_construction/comment_data/folds/<run>/comment_dev_data.csv'
test_data_path = 'dataset_construction/comment_data/folds/<run>/comment_test_data.csv'
train_data_path = 'dataset_construction/comment_data/folds/<run>/comment_train_data.csv'
val_data_path = 'dataset_construction/comment_data/folds/<run>/comment_val_data.csv'
train_os_data_path = 'dataset_construction/comment_data/folds/<run>/comment_oversampled_train_data.csv'
dev_os_df_path = 'dataset_construction/comment_data/folds/<run>/comment_oversampled_dev_data.csv'

dev_nos_embedding = 'dataset_construction/comment_data/folds/<run>/dev_nos_embedding.joblib'
dev_os_embedding = 'dataset_construction/comment_data/folds/<run>/dev_os_embedding.joblib'
train_nos_embedding = 'dataset_construction/comment_data/folds/<run>/train_nos_embedding.joblib'
train_os_embedding = 'dataset_construction/comment_data/folds/<run>/train_os_embedding.joblib'

train_test_data_directory = 'dataset/train_test_data'
train_test_comment_data = 'dataset_construction/comment_data/train_test_comment_data.json'
comment_data_result = "results/ml/fold/<run>/comment_data_results.csv"
sentence_data_result = "dataset_construction/sentence_data/sentence_data_result.csv"
comment_data_dl_result = "results/plm/comment_data_results.csv"
comment_data_llama_result = "dataset_construction/comment_data/llama/folds/<run>/mac_th/results.csv"
comment_data_llama_prediction = "results/predictions/llm/folds/<run>/"
ml_prediction_comment_dir = "results/predictions/ml"
ml_prediction_sentence_dir = "dataset_construction/sentence_data/predictions/"
lm_prediction_dir = "dataset_construction/comment_data/predictions/lm/"
lm_saved_model = "dataset_construction/comment_data/models/"

annotated_issues_id_list = [538721, 551837, 552914, 554061, 558970, 561168, 576837, 577462, 590389, 596726, 597389,
                            601912, 601999, 626855, 627984, 634654, 642412, 659018, 670853, 674446, 674609, 675961,
                            676248, 677173, 687929, 691184, 695213, 696748, 698552, 700508, 714547, 717147, 722137,
                            724586, 730907, 731836, 738440, 738759, 752781, 758103, 768901, 779500, 783505, 790547,
                            794507, 797889, 801993, 811773, 812431, 817341, 817531, 822952, 823917, 833964, 835157,
                            837955, 838565, 840284, 851828, 857034, 861246, 866470, 866474, 888630, 894646, 894931,
                            897027, 904571, 906912, 912496, 919434, 924397, 927544, 937475, 939475, 947523, 957093,
                            966240, 978610, 983489, 985257, 989204, 991812, 1003694, 1023280, 1025075, 1029919,
                            1033283, 1033887, 1047560, 1048721, 1055843, 1057903, 1066726, 1071367, 1074012,
                            1076026, 1080574, 1104875, 1105771, 1111046, 1116867, 1118597, 1118599, 1121826,
                            1130065, 1130253, 1132780, 1148078, 1173001, 1176028, 1184282, 1184945, 1187056,
                            1189924, 1190676, 1191113, 1203871, 1207931, 1209952, 1217192, 1217663, 1247539,
                            1252039, 1253516, 1253884, 1261576, 1262069, 1263083, 1268252, 1287522, 1297315,
                            1300206, 1306719, 1314193, 1315285, 1316964, 1319370, 1324053, 1324183, 1325297,
                            1325731, 1328511, 1329110, 1332070, 1339619, 1343787, 1344721, 1354406, 1357065,
                            1357386, 1359490, 1362590, 1373154, 1373249, 1382702, 1384677, 1385699, 1386502,
                            1390087, 1396319, 1401299, 1403319, 1405027, 1407955, 1410565, 1421170, 1430012,
                            1435264, 1435456, 1441779, 1454126, 1456911, 1467403, 1479989, 1482681, 1486218,
                            1487135, 1493860, 1494092, 1506200, 1509994, 1510786, 1514429, 1515665, 1521066,
                            1529006, 1537936, 1560574, 1569123, 1570673, 1571487, 1574259, 1581315, 1593658,
                            1598488, 1600320, 1614706, 1623400, 1624268, 1625850, 1629902, 1637897, 1640135,
                            1651332, 1651593, 1654383, 1662097, 1666607, 1677183, 1683093, 1686219, 1686238,
                            1688325, 1705253, 1705768, 1706916, 1716655, 1718748, 1732739, 1733898, 1742664,
                            1742770, 1748376, 1748902, 1751268, 1751721, 1751919, 1761435, 1761826, 1761994,
                            1762088, 1764716, 1769254, 1769748, 1774026, 1778644, 1790543, 1792203, 1794237,
                            1818468, 597071, 658675, 667586, 682449, 687754, 735312, 794101, 916390, 948882,
                            1073339, 1079905, 1092808, 1097236, 1132964, 1183934, 1203253, 1249818, 1254694,
                            1265066, 1271750, 1353954, 1355481, 1357049, 1401249, 1407435, 1421905, 1442559,
                            1455735, 1458856, 1462624, 1465254, 1469362, 1479309, 1485422, 1513854, 1519164,
                            1526439, 1565273, 1571472, 1576600, 1576778, 1579004, 1584273, 1615767, 1618477,
                            1630806, 1634650, 1644719, 1661727, 1689742, 1703558, 1705327, 1715838, 1717682,
                            1718031, 1762292, 1768744, 1769085, 1811466, 1815706, 819493, 1439571, 1096093, 593387,
                            552359, 1298588, 1179123, 1823751, 1340967, 1514062, 1093374, 855335, 1142350, 585832,
                            1728953, 1187966, 1809080, 1622369, 1753765, 1647930, 869069, 852135, 1347860, 686900,
                            1670911, 1486074, 1685379, 713597, 1516605, 1571124, 1821213, 660762, 1248726, 759033,
                            1645527, 1395751, 587434, 623742, 1820814, 1704164, 1024256, 1626111, 1794406, 1600017,
                            900384, 1363344, 874258, 712870, 1221593, 734506, 600489, 1528712, 1540794, 1079321,
                            1424993]
