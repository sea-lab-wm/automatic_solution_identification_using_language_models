import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain

from collections import Counter
from ktrain import text

# from utils import *

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
    E = 1   #TODO: Update epoch to 10
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