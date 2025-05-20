


import ast
import datetime
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy
import numpy as np
import pandas as pd
import ktrain
import pandas as pd
import joblib
from b_preprocess_data import text_preprocess

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from ktrain import text
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os
import pandas as pd
import joblib
from multiprocessing import Pool

from utils import *
from sources.utils import *


def train_evaluate(model_name, classes, X_train, y_train, X_val, BS, LR, E, model_save=None):
    set_seed(42)
    t_mod = text.Transformer(model_name, maxlen=256, class_names=classes)

    train = t_mod.preprocess_train(X_train, y_train)

    model = t_mod.get_classifier()
    learner = ktrain.get_learner(model, train_data=train, batch_size=BS)
    learner.set_weight_decay(0.01)
    learner.fit_onecycle(LR, E)

    predictor = ktrain.get_predictor(learner.model, preproc=t_mod)

    if model_save is not None:
        predictor.save(model_save)
        print(f"Model saved at: {model_save}")

    y_pred = predictor.predict(X_val)

    return y_pred

def start(run, model_name, config, prediction_path, result_path, model_path):
    dev_df = pd.read_csv(dev_data_path.replace('<run>', f'{run}'))
    val_df = pd.read_csv(val_data_path.replace('<run>', f'{run}'))
    dev_df_os = pd.read_csv(dev_os_df_path.replace('<run>', f'{run}'))

    model_name = model_name
    print(f"\n\n\nEvaluating Model: {model_name}\n\n\n")

    solution_col = 'label'
    oversample = config['oversample']

    if oversample:
        X_dev, y_dev = dev_df_os.drop(solution_col, axis=1), dev_df_os[solution_col]
    else:
        X_dev, y_dev = dev_df.drop(solution_col, axis=1), dev_df[solution_col]

    X_val, y_val = val_df.drop(solution_col, axis=1), val_df[solution_col]

    classes = [0, 1]

    X_dev = X_dev['text'].tolist()
    y_dev = y_dev.tolist()

    X_val = X_val['text'].tolist()
    y_val = y_val.tolist()

    print(f"Oversampling: {oversample}")
    print(f"Validation Data Size: {Counter(y_val)}")
    print(f"Whole Train Data Size: {Counter(y_dev)}")


    if len(X_dev) != len(y_dev):
        print("\n\n\nDev X and Y different size\n\n\n")
    if len(X_val) != len(y_val):
        print("\n\n\nVal X and Y different size\n\n\n")


    BS = config['batch_size']
    LR = config['learning_Rate']
    E = config['epoch']


    print("-" * 50)
    print(f"Batch Size:{BS}")
    print(f"Epoch:{E}")
    print(f"Learning Rate:{LR}")
    print("-" * 50)

    y_pred = train_evaluate(model_name, classes, X_dev, y_dev, X_val, BS, LR, E, model_save=model_path)

    print(f"Predictions :\n{y_pred}")

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_val)

    result = [run, oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN]        

    print("=" * 50)
    print(f"batch_sizes: {BS}")
    print(f"total_epochs: {E}")
    print(f"learning_rate: {LR}")
    print(f"Model name: {model_name}")
    print(f"Oversampling: {oversample}")
    print(f"Run: {run}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    print("=" * 50)

    append_row_to_csv(result_path, result)
    print(f"results appended to: {result_path}")

    test_df = test_df.copy()

    test_df[model_name + f"_{'os' if oversample else 'nos'}_prediction"] = y_pred
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    test_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved to: {prediction_path}")

if __name__ == "__main__":
    start()