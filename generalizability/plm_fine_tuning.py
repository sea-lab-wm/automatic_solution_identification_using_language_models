import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain
import pandas as pd
from collections import Counter
from ktrain import text
import tensorflow as tf
import random
import numpy as np

# from utils import *


def set_seed(seed):
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")


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


def train_model(model_name, classes, X_train, y_train, BS, LR, E, model_save):
    set_seed(42)
    tokenizer = text.Transformer(model_name, maxlen=256, class_names=classes)

    train = tokenizer.preprocess_train(X_train, y_train)

    model = tokenizer.get_classifier()

    learner = ktrain.get_learner(model, train_data=train, batch_size=BS)
    learner.set_weight_decay(0.01)
    learner.fit_onecycle(LR, E)

    predictor = ktrain.get_predictor(learner.model, preproc=tokenizer)

    if model_save is not None:
        predictor.save(model_save)
        print(f"Model saved at: {model_save}")


if __name__ == "__main__":
    # This is the best PLM
    model_name = "roberta-base"

    model_to_save_path = f"models/plm/full_data/{model_name}"
    mozilla_data_path = 'dataset/solution_identification_data/labeled_comment_data.csv'
    mozilla_data = pd.read_csv(mozilla_data_path)


    # Hyper-parameters of the best fold of PLM experiments
    oversampling = True
    BS = 16
    E = 10   #TODO: Update epoch to 10
    LR = 3e-5

    print(f"Training: {model_name}")

    if oversampling:
        mozilla_data = balance_classes(mozilla_data, 'label', output_type='df')

    classes = [0, 1]

    X_train = mozilla_data['text'].tolist()
    y_train = mozilla_data['label'].tolist()

    print(f"Oversampling: {oversampling}")
    print(f"Train Data Size: {Counter(y_train)}")

    if len(X_train) != len(y_train):
        print("\n\n\nTraining X and Y different size\n\n\n")

    print("-" * 50)
    print(f"Batch Size:{BS}")
    print(f"Epoch:{E}")
    print(f"Learning Rate:{LR}")
    print("-" * 50)

    train_model(model_name, classes, X_train, y_train, BS, LR, E, model_to_save_path)