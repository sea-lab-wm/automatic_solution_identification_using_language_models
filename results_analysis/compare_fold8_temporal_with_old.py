import glob

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

TEMPORAL_DIR = 'results/predictions/temporal-prediction'
MERGED_CSV = f'{TEMPORAL_DIR}/temporal_vs_old_predictions.csv'  # from compare_temporal_with_old_predictions.py
PREDICTIONS_CSV = f'{TEMPORAL_DIR}/fold8_temporal_vs_old_predictions.csv'
METRICS_CSV = f'{TEMPORAL_DIR}/fold8_temporal_vs_old_metrics.csv'

KEYS = ['issue_id', 'text_id']
FOLD = 8
MODELS = {'LLAMA-FT': 'Llama-3 FT', 'RoBERTa': 'RoBERTa', 'SVC': 'SVM'}
N_BOOTSTRAP = 2000


def load_probabilities():
    llm = pd.read_csv(glob.glob(f'{TEMPORAL_DIR}/llm/folds/{FOLD}/*_prediction.csv')[0])
    llm = llm[KEYS + ['positive_probability']].rename(columns={'positive_probability': 'temporal_LLAMA-FT_probability'})
    plm = pd.read_csv(f'{TEMPORAL_DIR}/plm/roberta-base/folds/{FOLD}/prediction_os_with_confidence.csv',
                      usecols=KEYS + ['roberta-base_os_confidence'])
    plm = plm.rename(columns={'roberta-base_os_confidence': 'temporal_RoBERTa_probability'})
    ml = pd.read_csv(f'{TEMPORAL_DIR}/ml/run_{FOLD}.csv', usecols=KEYS + [f'none_gpt_nos_SVC_{FOLD}_confidence'])
    ml = ml.rename(columns={f'none_gpt_nos_SVC_{FOLD}_confidence': 'temporal_SVC_probability'})
    return llm.merge(plm, on=KEYS, validate='1:1').merge(ml, on=KEYS, validate='1:1')


def counts(y, p):
    return (int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()),
            int(((y == 0) & (p == 0)).sum()), int(((y == 1) & (p == 0)).sum()))


def f1(y, p):
    tp, fp, _, fn = counts(y, p)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    merged = pd.read_csv(MERGED_CSV)
    data = merged[(merged['fold'] == FOLD) & merged['in_old']].reset_index(drop=True)
    data = data[KEYS + ['text', 'code', 'label'] + [f'{s}_{m}' for m in MODELS for s in ('old', 'temporal')]]
    data = data.merge(load_probabilities(), on=KEYS, how='left', validate='1:1')
    for m in MODELS:
        data[f'old_{m}'] = data[f'old_{m}'].astype(int)
    data.to_csv(PREDICTIONS_CSV, index=False)

    # bootstrap by issue: resample whole issues so comments of one discussion stay together
    rng = np.random.default_rng(42)
    issues = data['issue_id'].unique()
    rows_by_issue = {i: data.index[data['issue_id'] == i].to_numpy() for i in issues}
    samples = [np.concatenate([rows_by_issue[i] for i in rng.choice(issues, len(issues))]) for _ in range(N_BOOTSTRAP)]

    y = data['label'].to_numpy()
    rows = []
    for m, name in MODELS.items():
        old, new = data[f'old_{m}'].to_numpy(), data[f'temporal_{m}'].to_numpy()
        for setup, pred in (('old (random CV)', old), ('temporal', new)):
            tp, fp, tn, fn = counts(y, pred)
            boot = [f1(y[s], pred[s]) for s in samples]
            row = {'model': name, 'setup': setup, 'n': len(y), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                   'precision': tp / (tp + fp) if tp + fp else 0.0, 'recall': tp / (tp + fn) if tp + fn else 0.0,
                   'f1': f1(y, pred), 'f1_ci_low': np.percentile(boot, 2.5), 'f1_ci_high': np.percentile(boot, 97.5)}
            if setup == 'temporal':
                row['pr_auc'] = average_precision_score(y, data[f'temporal_{m}_probability'])
            rows.append(row)
        delta = [f1(y[s], new[s]) - f1(y[s], old[s]) for s in samples]
        old_only = int(((old == y) & (new != y)).sum())
        temporal_only = int(((old != y) & (new == y)).sum())
        rows.append({'model': name, 'setup': 'delta (temporal - old)', 'n': len(y),
                     'precision': rows[-1]['precision'] - rows[-2]['precision'],
                     'recall': rows[-1]['recall'] - rows[-2]['recall'],
                     'f1': rows[-1]['f1'] - rows[-2]['f1'],
                     'f1_ci_low': np.percentile(delta, 2.5), 'f1_ci_high': np.percentile(delta, 97.5),
                     'agreement': (old == new).mean(),
                     'mcnemar_old_only_correct': old_only, 'mcnemar_temporal_only_correct': temporal_only,
                     'mcnemar_p': binomtest(old_only, old_only + temporal_only).pvalue})
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False, float_format='%.4f')

    print(f'{len(data)} comments, {int(y.sum())} solutions, {len(issues)} issues')
    print(f'wrote {PREDICTIONS_CSV}\nwrote {METRICS_CSV}')


if __name__ == '__main__':
    main()
