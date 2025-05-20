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

def train_evaluate(model_name, classes, X_train, y_train, X_val, y_val, BS, LR, E, model_save=None):
    set_seed(42)
    t_mod = text.Transformer(model_name, maxlen=256, class_names=classes)

    train = t_mod.preprocess_train(X_train, y_train)
    val = t_mod.preprocess_test(X_val, y_val)

    model = t_mod.get_classifier()
    learner = ktrain.get_learner(model, train_data=train, batch_size=BS)
    learner.set_weight_decay(0.01)
    learner.fit_onecycle(LR, E)

    print(learner.validate(val_data=val, print_report=True, class_names=t_mod.get_classes()))

    predictor = ktrain.get_predictor(learner.model, preproc=t_mod)

    if model_save is not None:
        predictor.save(model_save)
        print(f"Model saved at: {model_save}")

    y_pred = predictor.predict(X_val)

    return y_pred

def evaluate_model_plm(run, mod, result_path, oversample=True, best_config={}):
    dev_df = pd.read_csv(dev_data_path.replace('<run>', f'{run}'))
    test_df = pd.read_csv(test_data_path.replace('<run>', f'{run}'))
    train_df = pd.read_csv(train_data_path.replace('<run>', f'{run}'))
    val_df = pd.read_csv(val_data_path.replace('<run>', f'{run}'))
    train_df_os = pd.read_csv(train_os_data_path.replace('<run>', f'{run}'))
    dev_df_os = pd.read_csv(dev_os_df_path.replace('<run>', f'{run}'))

    model_name = mod['model_name']
    print(f"\n\n\nEvaluating Model: {model_name}\n\n\n")

    solution_col = 'label'

    if oversample:
        X_train, y_train = train_df_os.drop(solution_col, axis=1), train_df_os[solution_col]
        X_dev, y_dev = dev_df_os.drop(solution_col, axis=1), dev_df_os[solution_col]
    else:
        X_train, y_train = train_df.drop(solution_col, axis=1), train_df[solution_col]
        X_dev, y_dev = dev_df.drop(solution_col, axis=1), dev_df[solution_col]

    X_val, y_val = val_df.drop(solution_col, axis=1), val_df[solution_col]
    X_test, y_test = test_df.drop(solution_col, axis=1), test_df[solution_col]

    classes = [0, 1]

    X_train = X_train['text'].tolist()
    y_train = y_train.tolist()

    X_dev = X_dev['text'].tolist()
    y_dev = y_dev.tolist()

    X_val = X_val['text'].tolist()
    y_val = y_val.tolist()

    X_test = X_test['text'].tolist()
    y_test = y_test.tolist()

    print(f"Oversampling: {oversample}")
    print(f"Train Data Size: {Counter(y_train)}")
    print(f"Test Data Size: {Counter(y_test)}")
    print(f"Validation Data Size: {Counter(y_val)}")
    print(f"Whole Train Data Size: {Counter(y_dev)}")
    
    if len(X_train) != len(y_train):
        print("\n\n\nTraining X and Y different size\n\n\n")
    if len(X_dev) != len(y_dev):
        print("\n\n\nDev X and Y different size\n\n\n")
    if len(X_val) != len(y_val):
        print("\n\n\nVal X and Y different size\n\n\n")
    if len(X_test) != len(y_test):
        print("\n\n\nTest X and Y different size\n\n\n")

    BS = -1
    LR = -1
    E = -1
    F1 = -1

    if not best_config:
        for bs in mod['params']['batch_size']:
            for lr in mod['params']['learning_rate']:
                for e in mod['params']['epoch']:
                    y_pred = train_evaluate(model_name, classes, X_train, y_train, X_val, y_val, bs, lr, e)

                    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_val)

                    print("=" * 50, "Inside Validation", "=" * 50)
                    print(f"Validation batch_sizes: {bs}")
                    print(f"Validation total_epochs: {e}")
                    print(f"Validation learning_rate: {lr}")
                    print(f"Validation Model name: {model_name}")
                    print(f"Validation Oversampling: {oversample}")
                    print(f"Validation Run: {run}")
                    print(f"Validation Accuracy: {accuracy}")
                    print(f"Validation Precision: {precision}")
                    print(f"Validation Recall: {recall}")
                    print(f"Validation F1 Score: {f1}")
                    print(f"Validation True Positives (TP): {TP}")
                    print(f"Validation False Positives (FP): {FP}")
                    print(f"Validation True Negatives (TN): {TN}")
                    print(f"Validation False Negatives (FN): {FN}")
                    print("=" * 50)

                    if f1 > F1:
                        BS = bs
                        LR = lr
                        E = e
                        F1 = f1
    else:
        BS = best_config['batch_size']
        LR = best_config['learning_Rate']
        E = best_config['epoch']


    print("-" * 50)
    print(f"Completed Validating with Max F1 Score: {F1}")
    print(f"Batch Size:{BS}")
    print(f"Epoch:{E}")
    print(f"Learning Rate:{LR}")
    print("-" * 50)

    model_path_folder = f"models/plm/{run}"
    model_file_name = f"{model_name}__BS_{BS}__LR_{LR}__E_{E}__Oversample_{'os' if oversample else 'nos'}.joblib"
    model_path = os.path.join(model_path_folder, model_file_name)

    y_pred = train_evaluate(model_name, classes, X_dev, y_dev, X_test, y_test, BS, LR, E, model_save=model_path)
    
    print(f"Predictions :\n{y_pred}")

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_pred, y_test)

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
    p = os.path.join(f"results/predictions/plm", model_name, f"folds/{run}", f"prediction_{'os' if oversample else 'nos'}.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    test_df.to_csv(p, index=False)
    print(f"Predictions saved to: {p}")

def dl_experiment(exp_config, res_path, run_best_model):
    runs = exp_config.get("eval_run", [])

    result = ["run", "oversample", "batch_size", "epoch", "learning_Rate", "model_name", "accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN"]
    append_row_to_csv(res_path, result)

    if not run_best_model:
        for run in runs:
            for model in exp_config['lm_model']:
                for oversampling in exp_config['oversampling']:
                    evaluate_model_plm(run, model, res_path, oversample=oversampling, best_config={})
    else:
        for run in runs:
            for model in exp_config['lm_model']:
                oversampling = exp_config['best_model_config'][str(run)][model['model_name']]['oversample']
                model_config = exp_config['best_model_config'][str(run)][model['model_name']]
                evaluate_model_plm(run, model, res_path, oversample=oversampling, best_config=model_config)

if __name__ == "__main__":
    exp_config = get_experiment_config()
    prepare_data(mapped_comment_data, train_test_comment_data)
    dl_experiment(exp_config, comment_data_dl_result, run_best_model=True)