import glob

import pandas as pd

TEMPORAL_DIR = 'results/predictions/temporal-prediction'
OLD_XLSX = 'results_analysis/ensembled_models.xlsx'
OLD_SHEET = 'Ensembled-Models'
RESULT_CSV = f'{TEMPORAL_DIR}/temporal_fold_results_common_pairs.csv'
PREDICTIONS_CSV = f'{TEMPORAL_DIR}/temporal_predictions_common_pairs.csv'

KEYS = ['issue_id', 'text_id']
N_FOLDS = 9
# model family in the results table -> short model name (as in ensembled_models.xlsx)
MODELS = {'MLM': 'SVC', 'PLM': 'RoBERTa', 'LLM': 'LLAMA-FT'}


def load_fold(fold):
    ml = pd.read_csv(f'{TEMPORAL_DIR}/ml/run_{fold}.csv')
    ml = ml[KEYS + ['text', 'code', 'label', f'none_gpt_nos_SVC_{fold}', f'none_gpt_nos_SVC_{fold}_confidence']]
    ml = ml.rename(columns={f'none_gpt_nos_SVC_{fold}': 'SVC_temporal',
                            f'none_gpt_nos_SVC_{fold}_confidence': 'SVC_temporal_probability'})

    # same predictions as prediction_os.csv, plus the positive-class probability
    plm = pd.read_csv(f'{TEMPORAL_DIR}/plm/roberta-base/folds/{fold}/prediction_os_with_confidence.csv',
                      usecols=KEYS + ['label', 'creation_time', 'comment_id',
                                      'roberta-base_os_prediction', 'roberta-base_os_confidence'])
    plm = plm.rename(columns={'label': 'label_plm', 'roberta-base_os_prediction': 'RoBERTa_temporal',
                              'roberta-base_os_confidence': 'RoBERTa_temporal_probability'})

    llm = pd.read_csv(glob.glob(f'{TEMPORAL_DIR}/llm/folds/{fold}/*_prediction.csv')[0])
    llm_col = [c for c in llm.columns if c.endswith('_prediction')][0]
    llm = llm[KEYS + ['label', llm_col, 'positive_probability']].rename(
        columns={'label': 'label_llm', llm_col: 'LLAMA-FT_temporal', 'positive_probability': 'LLAMA-FT_temporal_probability'})

    df = ml.merge(plm, on=KEYS, validate='1:1').merge(llm, on=KEYS, validate='1:1')
    assert len(df) == len(ml) == len(plm) == len(llm), f'fold {fold}: pair mismatch across model types'
    assert (df['label'] == df['label_plm']).all() and (df['label'] == df['label_llm']).all(), \
        f'fold {fold}: label mismatch across model types'
    df.insert(2, 'fold', fold)
    return df.drop(columns=['label_plm', 'label_llm'])


def load_old_pairs():
    old = pd.read_excel(OLD_XLSX, sheet_name=OLD_SHEET, header=1)
    # the sheet has metric summary rows below the predictions; keep only instance rows
    old = old[pd.to_numeric(old['issue_id'], errors='coerce').notna()
              & pd.to_numeric(old['text_id'], errors='coerce').notna()]
    old = old[KEYS + ['label'] + list(MODELS.values())].astype(int)
    old = old.rename(columns={'label': 'old_label', **{m: f'{m}_old' for m in MODELS.values()}})
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
    data = temporal.merge(load_old_pairs(), on=KEYS, how='inner', validate='1:1')
    assert (data['label'] == data['old_label']).all(), 'label mismatch between setups'
    data = data.drop(columns='old_label')
    data['comment_id'] = data['comment_id'].astype('Int64')
    data = data[KEYS + ['fold', 'comment_id', 'creation_time', 'text', 'code', 'label']
                + [c for m in MODELS.values() for c in (f'{m}_temporal', f'{m}_temporal_probability', f'{m}_old')]]
    data.to_csv(PREDICTIONS_CSV, index=False)

    rows = []
    for name in MODELS.values():
        groups = [(fold, df) for fold, df in data.groupby('fold')] + [('All', data)]  # All: pooled TP/FP/TN/FN
        for run, df in groups:
            rows.append({'model': name, 'run': run, **metrics(df['label'], df[f'{name}_temporal'])})
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)

    dropped = temporal.loc[~temporal.set_index(KEYS).index.isin(data.set_index(KEYS).index)]
    print(f'{len(data)} of {len(temporal)} temporal pairs kept; dropped per fold: '
          f'{dropped.groupby("fold").size().to_dict()}')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'wrote {PREDICTIONS_CSV}\nwrote {RESULT_CSV}')


if __name__ == '__main__':
    main()
