import glob

import pandas as pd

TEMPORAL_DIR = 'results/predictions/temporal-prediction'
OLD_XLSX = 'results_analysis/ensembled_models.xlsx'
OLD_SHEET = 'Ensembled-Models'
MERGED_CSV = f'{TEMPORAL_DIR}/temporal_vs_old_predictions.csv'
COMPARISON_CSV = f'{TEMPORAL_DIR}/temporal_vs_old_comparison.csv'

KEYS = ['issue_id', 'text_id']
N_FOLDS = 9
# temporal model -> column name of the same model in the old sheet
TEMPORAL_MODELS = {'LLAMA-FT': 'LLAMA-FT', 'RoBERTa': 'RoBERTa', 'SVC': 'SVC'}
OLD_MODELS = ['LLAMA-FT', 'RoBERTa', 'BERT', 'SVC', 'XLNet', 'DistilBERT', 'Logistic_Regression',
              'Random_Forest', 'K_Nearest_Neighbors', 'Decision_Tree', 'Llama-Prompt', 'Naive_Bayes']


def load_temporal():
    folds = []
    for fold in range(N_FOLDS):
        ml = pd.read_csv(f'{TEMPORAL_DIR}/ml/run_{fold}.csv')
        ml = ml[KEYS + ['text', 'code', 'label', f'none_gpt_nos_SVC_{fold}']]
        ml = ml.rename(columns={f'none_gpt_nos_SVC_{fold}': 'temporal_SVC'})

        llm_file = glob.glob(f'{TEMPORAL_DIR}/llm/folds/{fold}/*_prediction.csv')[0]
        llm = pd.read_csv(llm_file)
        llm_col = [c for c in llm.columns if c.endswith('_prediction')][0]
        llm = llm[KEYS + [llm_col]].rename(columns={llm_col: 'temporal_LLAMA-FT'})

        plm = pd.read_csv(f'{TEMPORAL_DIR}/plm/roberta-base/folds/{fold}/prediction_os.csv',
                          usecols=KEYS + ['roberta-base_os_prediction'])
        plm = plm.rename(columns={'roberta-base_os_prediction': 'temporal_RoBERTa'})

        df = ml.merge(llm, on=KEYS, how='inner', validate='1:1').merge(plm, on=KEYS, how='inner', validate='1:1')
        assert len(df) == len(ml) == len(llm) == len(plm), f'fold {fold}: pair mismatch across model types'
        df.insert(2, 'fold', fold)
        folds.append(df)
    temporal = pd.concat(folds, ignore_index=True)
    assert not temporal.duplicated(KEYS).any(), 'duplicate pairs across temporal folds'
    return temporal


def load_old():
    old = pd.read_excel(OLD_XLSX, sheet_name=OLD_SHEET, header=1)
    # the sheet has metric summary rows below the predictions; keep only instance rows
    old = old[pd.to_numeric(old['issue_id'], errors='coerce').notna()
              & pd.to_numeric(old['text_id'], errors='coerce').notna()].copy()
    old[KEYS] = old[KEYS].astype(int)
    old['label'] = old['label'].astype(int)
    old[OLD_MODELS] = old[OLD_MODELS].astype(int)
    assert not old.duplicated(KEYS).any(), 'duplicate pairs in old predictions'
    return old


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'n': len(y_true), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': (tp + tn) / len(y_true)}


def main():
    temporal = load_temporal()
    old = load_old()

    old_cols = old[KEYS + ['label'] + OLD_MODELS].rename(
        columns={'label': 'old_label', **{m: f'old_{m}' for m in OLD_MODELS}})
    merged = temporal.merge(old_cols, on=KEYS, how='left', validate='1:1')
    merged.insert(6, 'in_old', merged['old_label'].notna())
    common = merged[merged['in_old']]
    assert (common['label'] == common['old_label']).all(), 'label mismatch between setups'
    merged = merged.drop(columns='old_label')
    old_pred_cols = [f'old_{m}' for m in OLD_MODELS]
    merged[old_pred_cols] = merged[old_pred_cols].astype('Int64')
    merged.to_csv(MERGED_CSV, index=False)

    rows = []
    for model, old_model in TEMPORAL_MODELS.items():
        rows.append({'setup': 'temporal', 'model': model, 'instances': 'all temporal pairs',
                     **metrics(merged['label'], merged[f'temporal_{model}'])})
    for model, old_model in TEMPORAL_MODELS.items():
        old_m = metrics(common['label'], common[f'old_{old_model}'])
        new_m = metrics(common['label'], common[f'temporal_{model}'])
        agreement = (common[f'old_{old_model}'] == common[f'temporal_{model}']).mean()
        rows.append({'setup': 'old', 'model': model, 'instances': 'common pairs', **old_m})
        rows.append({'setup': 'temporal', 'model': model, 'instances': 'common pairs', **new_m,
                     'delta_precision': new_m['precision'] - old_m['precision'],
                     'delta_recall': new_m['recall'] - old_m['recall'],
                     'delta_f1': new_m['f1'] - old_m['f1'],
                     'agreement_with_old': agreement})
    for model in OLD_MODELS:
        if model in TEMPORAL_MODELS.values():
            continue
        rows.append({'setup': 'old', 'model': model, 'instances': 'common pairs',
                     **metrics(common['label'], common[f'old_{model}'])})
    for model in OLD_MODELS:
        rows.append({'setup': 'old', 'model': model, 'instances': 'all old pairs',
                     **metrics(old['label'], old[model])})
    pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False, float_format='%.6f')

    print(f'temporal pairs: {len(merged)}, common with old: {len(common)}, '
          f'temporal-only: {len(merged) - len(common)}, old-only (discarded): {len(old) - len(common)}')
    print(f'wrote {MERGED_CSV}\nwrote {COMPARISON_CSV}')


if __name__ == '__main__':
    main()
