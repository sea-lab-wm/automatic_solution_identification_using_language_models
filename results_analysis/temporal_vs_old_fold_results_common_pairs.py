import glob

import pandas as pd

TEMPORAL_DIR = 'results/predictions/temporal-prediction'
OLD_XLSX = 'results_analysis/ensembled_models.xlsx'
OLD_SHEET = 'Ensembled-Models'
RESULT_CSV = f'{TEMPORAL_DIR}/temporal_vs_old_fold_results_common_pairs.csv'
# one wide table per metric with columns <model>_<metric>_old, <model>_<metric>_temporal
WIDE_CSV = f'{TEMPORAL_DIR}/temporal_vs_old_fold_{{}}_common_pairs.csv'
WIDE_METRICS = ['precision', 'recall', 'f1']

KEYS = ['issue_id', 'text_id']
N_FOLDS = 9
MODELS = ['SVC', 'RoBERTa', 'LLAMA-FT']  # column names in ensembled_models.xlsx


def load_fold(fold):
    ml = pd.read_csv(f'{TEMPORAL_DIR}/ml/run_{fold}.csv', usecols=KEYS + ['label', f'none_gpt_nos_SVC_{fold}'])
    ml = ml.rename(columns={f'none_gpt_nos_SVC_{fold}': 'SVC_temporal'})

    plm = pd.read_csv(f'{TEMPORAL_DIR}/plm/roberta-base/folds/{fold}/prediction_os.csv',
                      usecols=KEYS + ['label', 'roberta-base_os_prediction'])
    plm = plm.rename(columns={'label': 'label_plm', 'roberta-base_os_prediction': 'RoBERTa_temporal'})

    llm = pd.read_csv(glob.glob(f'{TEMPORAL_DIR}/llm/folds/{fold}/*_prediction.csv')[0])
    llm_col = [c for c in llm.columns if c.endswith('_prediction')][0]
    llm = llm[KEYS + ['label', llm_col]].rename(columns={'label': 'label_llm', llm_col: 'LLAMA-FT_temporal'})

    df = ml.merge(plm, on=KEYS, validate='1:1').merge(llm, on=KEYS, validate='1:1')
    assert len(df) == len(ml) == len(plm) == len(llm), f'fold {fold}: pair mismatch across model types'
    assert (df['label'] == df['label_plm']).all() and (df['label'] == df['label_llm']).all(), \
        f'fold {fold}: label mismatch across model types'
    df.insert(2, 'fold', fold)
    return df.drop(columns=['label_plm', 'label_llm'])


def load_old():
    old = pd.read_excel(OLD_XLSX, sheet_name=OLD_SHEET, header=1)
    # the sheet has metric summary rows below the predictions; keep only instance rows
    old = old[pd.to_numeric(old['issue_id'], errors='coerce').notna()
              & pd.to_numeric(old['text_id'], errors='coerce').notna()]
    old = old[KEYS + ['label'] + MODELS].astype(int)
    old = old.rename(columns={'label': 'old_label', **{m: f'{m}_old' for m in MODELS}})
    assert not old.duplicated(KEYS).any(), 'duplicate pairs in old predictions'
    return old


def metrics(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': round(precision * 100, 2), 'recall': round(recall * 100, 2), 'f1': round(f1 * 100, 2),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def main():
    temporal = pd.concat([load_fold(fold) for fold in range(N_FOLDS)], ignore_index=True)
    assert not temporal.duplicated(KEYS).any(), 'duplicate pairs across temporal folds'

    # keep only pairs that also exist in the old predictions (ensembled_models.xlsx)
    data = temporal.merge(load_old(), on=KEYS, how='inner', validate='1:1')
    assert (data['label'] == data['old_label']).all(), 'label mismatch between setups'

    rows = []
    for model in MODELS:
        # old predictions are scored on the same comments, grouped by temporal fold
        groups = [(fold, df) for fold, df in data.groupby('fold')] + [('All', data)]  # All: pooled TP/FP/TN/FN
        for run, df in groups:
            old = metrics(df['label'], df[f'{model}_old'])
            new = metrics(df['label'], df[f'{model}_temporal'])
            rows.append({'model': model, 'run': run,
                         **{f'{k}_{setup}': m[k] for k in old for setup, m in (('old', old), ('temporal', new))}})
    result = pd.DataFrame(rows)
    result.to_csv(RESULT_CSV, index=False)

    for metric in WIDE_METRICS:
        wide = result.pivot(index='run', columns='model', values=[f'{metric}_old', f'{metric}_temporal'])
        columns = [(f'{metric}_{setup}', m) for m in MODELS for setup in ('old', 'temporal')]
        wide = wide.loc[result['run'].unique(), columns]
        wide.columns = [f'{m}_{col}' for col, m in columns]
        wide.to_csv(WIDE_CSV.format(metric), index_label='run')

    print(f'{len(data)} of {len(temporal)} temporal pairs kept')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'wrote {RESULT_CSV}')
    for metric in WIDE_METRICS:
        print(f'wrote {WIDE_CSV.format(metric)}')


if __name__ == '__main__':
    main()
