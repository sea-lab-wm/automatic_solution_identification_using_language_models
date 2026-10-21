import os
from utils import append_row_to_csv, calculate_results, set_seed
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ktrain
from ktrain import text
import os
import pandas as pd
# import tensorflow as tf
# import gc
# from tensorflow.keras import backend as K

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)


def get_predictions(X_test, predictor):
    # predictor = ktrain.load_predictor(model_path)
    y_pred = predictor.predict(X_test)
    return y_pred


def fine_tune_model(X_train, y_train, predictor, BS, LR, E, model_to_save_path):
    model = predictor.model
    tokenizer_model = predictor.preproc

    train = tokenizer_model.preprocess_train(X_train, y_train)
    learner = ktrain.get_learner(
        model,
        train_data=train,
        batch_size=BS
    )
    learner.fit_onecycle(LR, E)
    predictor = ktrain.get_predictor(learner.model, preproc=tokenizer_model)

    predictor.save(model_to_save_path)
    print(f"Model saved at: {model_to_save_path}")
    # del learner
    # del predictor
    # del model
    # del tokenizer_model
    # del train


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


def load_base_model(model_name):
    set_seed(42)
    preprocessor = text.Transformer(model_name, maxlen=256, classes=[0, 1])
    model = preprocessor.get_classifier()

    predictor = ktrain.get_predictor(model, preproc=preprocessor)
    return predictor


if __name__ == "__main__":
    # Best ML model
    model_name = "roberta-base"
    fine_tuned_model_path = f"generalizability/models/plm/roberta-base/mozilla/roberta-base-ft-mozilla"
    # Best configuration for roberta-base of Mozilla experiments
    oversample = False
    BS = 16
    LR = 3.00E-05
    E = 10

    # We experiment with the datasets of two new projects
    projects = ['Gnucash', 'Chromium']
    # projects = ['Chromium']

    # On these two datasets, we evaluate the performance of three trained models:
        # 1. mozilla: Model trained on the Mozilla data
        # 2. project: Model trained on the project (i.e., GunCash or Chromium) data
        # 3. mozilla-project: Model trained on both Mozilla and the project (i.e., GunCash or Chromium) data
    # exp_types = ['mozilla', 'project', 'mozilla-project']
    exp_types = ['mozilla-project']

    for project in projects:
        for exp_type in exp_types:
            if exp_type == 'mozilla':
                exp_name = "RoBERTa-FT-Mozilla"
            elif exp_type == 'project':
                exp_name = f"RoBERTa-FT-{project}"
            elif exp_type == 'mozilla-project':
                exp_name = f"RoBERTa-FT-Mozilla-{project}"

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

            # Load the fine-tuned model on Mozilla dataset
            predictor = ktrain.load_predictor(fine_tuned_model_path)

            # Run experiment 1
            if exp_type == 'mozilla':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                y_pred = get_predictions(X_test_project, predictor)
                y_preds.extend(y_pred)
                y_vals.extend(y_test_project)

                project_data[exp_name] = y_pred
                prediction_df = pd.concat([prediction_df, project_data], ignore_index=True)

            # Run experiment 2 or 3
            elif exp_type == 'project' or exp_type == 'mozilla-project':
                print(f"\n\n{exp_name} on {project} dataset:")
                print("===============================================")
                folds = prepare_dataset(project_data)
                for i, fold in enumerate(folds):
                    print(f"Fold {i + 1}/{len(folds)}")

                    model_to_save_path_new = f"{model_to_save_path}/{i}"

                    train_issue_id = fold['train']
                    test_issue_id = fold['test']
                    train_df = project_data[project_data['issue_id'].isin(train_issue_id)]
                    test_df = project_data[project_data['issue_id'].isin(test_issue_id)]

                    X_train = train_df['text'].tolist()
                    y_train = train_df['label'].tolist()

                    X_test = test_df['text'].tolist()
                    y_test = test_df['label'].tolist()

                    # Load the base model
                    if exp_type == 'project':
                        predictor = load_base_model(model_name)

                    fine_tune_model(X_train, y_train, predictor, BS, LR, E, model_to_save_path_new)

                    # Load the new fine-tuned model
                    predictor = ktrain.load_predictor(model_to_save_path_new)
                    y_pred = get_predictions(X_test, predictor)

                    y_vals.extend(y_test)
                    y_preds.extend(y_pred)

                    test_df = test_df.copy()
                    test_df[exp_name] = y_pred

                    prediction_df = pd.concat([prediction_df, test_df], ignore_index=True)

                    # # Clear memory after training
                    # del predictor
                    # K.clear_session()
                    # gc.collect()

            FN, FP, TN, TP, accuracy, f1, precision, recall = calculate_results(y_preds, y_vals)
            result = [oversample, BS, E, LR, model_name, accuracy, precision, recall, f1, TP, FP, TN, FN]
            result_df = pd.concat([result_df, pd.DataFrame([result], columns=result_df.columns)], ignore_index=True)

            print("=" * 50)
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print("=" * 50)

            # Save results
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            result_df.to_csv(result_path, index=False)
            # Save predictions
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            prediction_df.to_csv(prediction_path, index=False)
            print(f"Predictions saved to: {prediction_path}")
