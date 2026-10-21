import os
import gc
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
from utils import append_row_to_csv, calculate_results, set_seed
import ktrain

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Memory growth setup for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def run_fold(i, fold, exp_type, model_name, project, project_data_path, BS, LR, E, fine_tuned_model_path, model_to_save_path):
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"  # <- required in subprocess too
    import tensorflow as tf
    import gc
    from tensorflow.keras import backend as K
    import ktrain
    from ktrain import text
    from utils import set_seed

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    set_seed(42)
    project_data = pd.read_csv(project_data_path)

    train_issue_id = fold['train']
    test_issue_id = fold['test']
    train_df = project_data[project_data['issue_id'].isin(train_issue_id)]
    test_df = project_data[project_data['issue_id'].isin(test_issue_id)]

    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()

    model_to_save_path_new = f"{model_to_save_path}/{i}"

    if exp_type == 'project':
        preprocessor = text.Transformer(model_name, maxlen=256, classes=[0, 1])
        model = preprocessor.get_classifier()
        predictor = ktrain.get_predictor(model, preproc=preprocessor)
    elif exp_type == 'mozilla-project':
        predictor = ktrain.load_predictor(fine_tuned_model_path)

    train = predictor.preproc.preprocess_train(X_train, y_train)
    learner = ktrain.get_learner(predictor.model, train_data=train, batch_size=BS)
    learner.fit_onecycle(LR, E)

    predictor = ktrain.get_predictor(learner.model, preproc=predictor.preproc)
    predictor.save(model_to_save_path_new)

    del learner
    del predictor
    del train
    K.clear_session()
    gc.collect()

def prepare_dataset(data):
    issue_ids = data['issue_id'].unique()
    splits = []
    for i in range(len(issue_ids)):
        test_ids = [issue_ids[i]]
        train_ids = [id_ for j, id_ in enumerate(issue_ids) if j != i]
        splits.append({'train': train_ids, 'test': test_ids})
    return splits

def get_predictions(X_test, predictor):
    return predictor.predict(X_test)

if __name__ == "__main__":
    model_name = "roberta-base"
    fine_tuned_model_path = f"generalizability/models/plm/roberta-base/mozilla/roberta-base-ft-mozilla"
    oversample = False
    BS = 16
    LR = 3.00E-05
    E = 10

    projects = ['gnucash', 'chromium']
    exp_types = ['mozilla', 'project', 'mozilla-project']

    for project in projects:
        for exp_type in exp_types:
            if exp_type == 'mozilla':
                exp_name = "ft_mozilla"
            elif exp_type == 'project':
                exp_name = f"ft_{project}"
            elif exp_type == 'mozilla-project':
                exp_name = f"ft_mozilla_{project}"

            data_path = f"generalizability/dataset/{project}.csv"
            model_to_save_path = f"generalizability/models/plm/{model_name}/{project}/{exp_name}"
            prediction_path = f"generalizability/predictions/plm/{model_name}/{project}/{exp_name}_predictions.csv"
            result_path = f"generalizability/results/plm/{model_name}/{project}/{exp_name}_results.csv"

            project_data = pd.read_csv(data_path)
            X_test_project = project_data['text'].tolist()
            y_test_project = project_data['label'].tolist()

            y_vals = []
            y_preds = []
            result_df = pd.DataFrame(columns=['Oversample', 'Batch_Size', 'Epochs', 'LR', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'TP', 'FP', 'TN', 'FN'])
            prediction_df = pd.DataFrame()

            if exp_type == 'mozilla':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("==============================================")
                predictor = ktrain.load_predictor(fine_tuned_model_path)
                y_pred = get_predictions(X_test_project, predictor)
                y_preds.extend(y_pred)
                y_vals.extend(y_test_project)
                del predictor
            else:
                print(f"\n\n{exp_name} on {project} dataset:")
                print("==============================================")
                folds = prepare_dataset(project_data)
                for i, fold in enumerate(folds):
                    p = mp.Process(target=run_fold, args=(i, fold, exp_type, model_name, project, data_path, BS, LR, E, fine_tuned_model_path, model_to_save_path))
                    p.start()
                    p.join()

                    predictor = ktrain.load_predictor(f"{model_to_save_path}/{i}")
                    test_df = project_data[project_data['issue_id'].isin(fold['test'])].copy()
                    X_test = test_df['text'].tolist()
                    y_test = test_df['label'].tolist()
                    y_pred = get_predictions(X_test, predictor)

                    y_vals.extend(y_test)
                    y_preds.extend(y_pred)
                    test_df[model_name] = y_pred
                    prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

                    del predictor
                    tf.keras.backend.clear_session()
                    gc.collect()

            FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
            result = [oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN]
            result_df = pd.concat([result_df, pd.DataFrame([result], columns=result_df.columns)], ignore_index=True)

            print("=" * 50)
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print("=" * 50)

            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_df.to_csv(result_path, index=False)
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            prediction_df.to_csv(prediction_path, index=False)
            print(f"Predictions saved to: {prediction_path}")
