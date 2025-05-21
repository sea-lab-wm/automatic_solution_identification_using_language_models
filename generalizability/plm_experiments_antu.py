import os
from utils import append_row_to_csv, calculate_results, set_seed
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
import ktrain
import pandas as pd
from collections import Counter
from ktrain import text
import os
import pandas as pd
import joblib


def train_evaluate(predictor, classes, X_train, y_train, X_test, BS, LR, E, model_save=None):
    # set_seed(42)
    # t_mod = text.Transformer(model_name, maxlen=256, class_names=classes)
    model   = predictor.model
    t_mod = predictor.preproc
    train = t_mod.preprocess_train(X_train, y_train)
    learner = ktrain.get_learner(
        model,
        train_data=train,
        batch_size=BS
    )
    learner.fit_onecycle(LR, E)
    predictor = ktrain.get_predictor(learner.model, preproc=t_mod)
    if model_save is not None:
        predictor.save(model_save)
        print(f"Model saved at: {model_save}")
    y_pred = predictor.predict(X_test)
    return y_pred


def start(X_dev, y_dev, X_val, y_val, model_path, predictor_model=None):
    # dev_df = pd.read_csv(dev_data_path.replace('<run>', f'{run}'))
    # val_df = pd.read_csv(val_data_path.replace('<run>', f'{run}'))
    # dev_df_os = pd.read_csv(dev_os_df_path.replace('<run>', f'{run}'))
    # model_name = model_name
    # print(f"\n\n\nEvaluating Model: {model_name}\n\n\n")
    # solution_col = 'label'
    # oversample = config['oversample']
    # if oversample:
    #     X_dev, y_dev = dev_df_os.drop(solution_col, axis=1), dev_df_os[solution_col]
    # else:
    #     X_dev, y_dev = dev_df.drop(solution_col, axis=1), dev_df[solution_col]
    # X_val, y_val = val_df.drop(solution_col, axis=1), val_df[solution_col]
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

    print("-" * 50)
    print(f"Batch Size:{BS}")
    print(f"Epoch:{E}")
    print(f"Learning Rate:{LR}")
    print("-" * 50)
    y_pred = train_evaluate(predictor_model, classes, X_dev, y_dev, X_val, BS, LR, E, model_save=model_path)
    print(f"Predictions :\n{y_pred}")
    print(f"Validation Data Size:\n{y_val}")

    return y_pred


def prepare_dataset(data):
    issue_ids = data['issue_id'].unique()
    # print(f"Unique issue IDs: {len(issue_ids)}")
    # print(issue_ids)
    splits = []
    for i in range(len(issue_ids)):
        test_ids = [issue_ids[i]]
        train_ids = [id_ for j, id_ in enumerate(issue_ids) if j != i]
        splits.append({'train': train_ids, 'test': test_ids})
    # # Example: print the splits
    # for idx, split in enumerate(splits):
    #     print(f"Split {idx+1}:")
    #     print(f"  Train IDs: {split['train']}")
    #     print(f"  Test ID: {split['test']}")
    return splits


def load_model(model_name=None, model_path = None):
    if not model_path is None:
        return ktrain.load_predictor(model_path)

    if not model_name is None:
        set_seed(42)

        preprocessor = text.Transformer(model_name, maxlen=256)
        model = preprocessor.get_classifier()

        predictor = ktrain.get_predictor(model, preproc=preprocessor)

        return predictor

    return None



if __name__ == "__main__":
    data = pd.read_csv("generalizability/dataset/GnuCash_data.csv")
    fine_tuned_model_path = f"models/plm/0/roberta-base__BS_32__LR_3e-05__E_10__Oversample_nos.joblib"
    prediction_path = f"generalizability/predictions/plm/plm_predictions.csv"
    result_path = "generalizability/results/plm_results_new.csv"

    model_names = [
        'roberta-base'
    ]
    config = {
        "batch_size": 16,
        "learning_Rate": 3.00E-05,
        "epoch": 10,
        "oversample": False
    }

    oversample = config['oversample']
    BS = config['batch_size']
    LR = config['learning_Rate']
    E = config['epoch']

    y_vals = []
    y_preds = []
    result_df = pd.DataFrame(columns=['Oversample', 'Batch_Size', 'Epochs', 'LR', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TP', 'FP', 'TN', 'FN'])
    prediction_df = pd.DataFrame()

    folds = prepare_dataset(data)
    # print(folds)
    for i, fold in enumerate(folds):
        # if i>2:
        #     break
        print(f"Fold {i+1}/{len(folds)}")
        for model_name in model_names:
            model_to_save_path = f"generalizability/models/plm/{i}/{model_name}"

            train_issue_id = fold['train']
            test_issue_id = fold['test']
            # print("Fold:", fold)
            train_df = data[data['issue_id'].isin(train_issue_id)]
            test_df = data[data['issue_id'].isin(test_issue_id)]
            X_dev = train_df
            y_dev = train_df['label']
            X_val = test_df
            y_val = test_df['label']
            # print(f"Train Data Size: {len(X_val)}")
            # print(f"Test Data Size: {len(y_val)}")
            model = load_model(fine_tuned_model_path)

            if model is None:
                print("Model not loaded")
                return

            y_pred = start(X_dev, y_dev, X_val, y_val, model_to_save_path, predictor_model=model)

            y_vals.append(y_val)
            y_preds.append(y_pred)

            test_df = test_df.copy()
            test_df[model_name] = y_pred

            prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

    FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
    result = [oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN]
    result_df = pd.concat([result_df, pd.DataFrame([result], columns=result_df.columns)], ignore_index=True)

    print("=" * 50)
    print(f"batch_sizes: {BS}")
    print(f"total_epochs: {E}")
    print(f"learning_rate: {LR}")
    print(f"Model name: {model_name}")
    print(f"Oversampling: {oversample}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")

    result_df.to_csv(result_path, index=False)

    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    prediction_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved to: {prediction_path}")