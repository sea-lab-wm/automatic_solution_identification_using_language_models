# Replication Package

This is the replication package of the paper **"Evaluating Language Model Applications for Identifying Solution-Related Content in Issue Report Discussions: A Case Study of Firefox"** (Empirical Software Engineering).

The study classifies each comment of 356 Mozilla Firefox issue reports as *solution* or *non-solution*, comparing three applications of language models: embeddings for traditional machine learning (ML) models, prompting, and fine-tuning. It covers six ML models, four pre-trained language models (PLMs), and Llama-3, plus ensembles, a temporal evaluation, and a transferability study on Chromium and GnuCash.

The package contains the dataset, the source code, the configurations, the predictions and results of the models, and the analyses reported in the paper.

## Contents

| Paper | Artifacts |
|---|---|
| Dataset (Sections 3.2-3.3) | `dataset/`, `annotation_catalog/` |
| RQ1: ML models with LM embeddings (Table 4) | `sources/b_*`, `sources/c_*`, `sources/d_mlm_experiments.py`, `experiments_table4.json`, `results/ml/`, `results/rerun-ml-table4/` |
| RQ2: Prompting (Table 5) | `sources/f_*`, `sources/g_*`, `prompting/`, `results/llm_prompting/` |
| RQ3: Fine-tuning (Table 6) | `sources/e_plm_experiments.py`, `sources/h_llm_fine_tuning_experiments.py`, `experiments_best.json`, `results/plm/`, `results/llm_fine_tuning/` |
| RQ4: Ensembles (Table 7) | `results_analysis/ensembled_models.xlsx`, `results_analysis/complexity_analysis.md` |
| RQ5: Issue types and problem categories | `results_analysis/issue_type_and_category_analysis.xlsx` |
| RQ6-RQ7: Qualitative analysis of the errors | `results_analysis/create_venn_diagram.py`, `results_analysis/select_random_sample.py`, `results_analysis/compute_agreement_index.py` |
| Temporal evaluation (Section 4.7, Tables 8-9) | `sources/i_temporal_experiment.py`, `sources/temp/add_datetime_to_labeled_data.py`, `dataset/solution_identification_data/temporal_folds/`, `results/temporal-results/`, `results/predictions/temporal-prediction/`, `results_analysis/compare_temporal_with_old_predictions.py`, `results_analysis/temporal_vs_old_fold_results_common_pairs.py` |
| Explanations of the findings (Sections 4.1-4.4) | `results_analysis/explain_findings.py`, `results_analysis/plot_error_overlap_venn.py`, `results_analysis/generated/explain_findings/`, `results_analysis/generated/error_overlap/` |
| Statistical significance (Section 3.6, Table 10) | `results_analysis/significance_tests.py`, `results_analysis/significance_table.py`, `results_analysis/generated/significance_tests/` |
| RQ8: Transferability study (Section 5, Table 12) | `generalizability/`, `experiments.json` |

## Directory Structure

### `dataset/`
- `annotation_data/annotation_data.json`: the annotated text snippets of the 356 Firefox issues from our prior work, with the annotation code assigned to each snippet.
- `issue_data/`: the title, description, comments (`issue_title_and_comments.json`), and metadata (`issue_meta_data.csv`) of the 356 issues.
- `solution_identification_data/`:
  - `labeled_comment_data.csv`: the 4,917 comments, each labeled as solution (1) or non-solution (0), with its annotation codes.
  - `manual_comment_mapping.csv`: the three comments whose annotations were mapped to the issue comments manually.
  - `prompt_set.csv` and `train_test_split_prompt_set.json`: the 50 comments used for prompt development.
  - `train_test_split.json`: the 10 folds of the cross-validation over the remaining 4,867 comments.
  - `temporal_folds/fold_<k>.csv`: the 10 chronological folds of the temporal evaluation (comments sorted by creation date), with their `text-embedding-3-large` embeddings.

### `annotation_catalog/`
- `issue_annotation.xlsx`: the annotation code catalog (`annotation_codes` sheet, with the definition, rules, and examples of each code) and the problem category catalog (`problem_categories` sheet) of our prior work. Copied from its replication package: https://zenodo.org/records/14860151. A comment is a solution comment if it contains a snippet annotated with `POTENTIAL_SOLUTION_DESIGN`, `SOLUTION_REVIEW`, or `CODE_IMPLEMENTATION`.

### `sources/`
Scripts for data preparation, training, and evaluation, run in this order:
1. `a_annotation_and_issue_data_mapping.py`: maps the annotations to the issue comments and labels each comment.
2. `b_preprocess_data.py`: preprocesses the comments for the Tf-Idf features (lowercasing, lemmatization, punctuation and stop word removal).
3. `c_generate_embeddings.py`: generates the Bert, Llama, and Gpt (OpenAI `text-embedding-3-large`) embeddings of the comments. Requires an OpenAI API key and an Ollama server.
4. `d_mlm_experiments.py`: trains and tests the six ML models (RQ1).
5. `e_plm_experiments.py`: fine-tunes and tests the PLMs with ktrain (RQ3).
6. `f_llm_prompting_experiments.py`: generates the prompts and collects the Llama-3 responses through Ollama (RQ2).
7. `g_compute_metrics_for_prompting.py`: computes the precision, recall, and F1 score of the prompting experiments.
8. `h_llm_fine_tuning_experiments.py`: fine-tunes Llama-3 8B with LoRA and 4-bit quantization and tests it (RQ3).
9. `i_temporal_experiment.py`: builds the training and test sets of the nine executions of the temporal evaluation from `temporal_folds/fold_<k>.csv` (expanding window). `temp/add_datetime_to_labeled_data.py` adds the comment creation dates used to sort the comments.
- `utils.py`: shared paths and helper functions.

### Configuration files
- `experiments_best.json`: models and hyperparameter grids, and the selected PLM and Llama-3 configurations, used by `sources/`.
- `experiments_table4.json`: the 48 ML configurations of Table 4 (6 models x 4 text representations x with/without Smote) with the hyperparameters selected in each fold.
- `experiments.json`: configuration used by the transferability scripts in `generalizability/`.

### `prompting/`
- `prompt_templates.csv`: the prompt templates (zero-shot, few-shot, and chain-of-thought, with and without the issue description). Prompt versions `1.1.*`, `1.2.*`, and `1.3.*` are the zero-shot, few-shot, and chain-of-thought prompts; the third number is 1 when the prompt includes the issue description; the last number is 3 (zero-shot, or 4 examples) or 4 (7 examples).
- `generated_prompts_dev_set/`, `generated_responses_dev_set/`: prompts and responses on the prompt development set.
- `generated_prompts_test_set/run<k>/`: prompts for the test set of each fold (`run1`-`run10` correspond to folds 0-9).
- `generated_responses_test_set/fold<k>/`: the Llama-3 responses for each fold and prompt (three executions per prompt, columns `llama_prediction_0`-`2`), after labeling the non-binary responses (Section 4.2.2).

### `results/`
- `ml/comment_data_results copy.csv`: results of the 48 ML configurations in each fold, with the hyperparameters selected by the nested cross-validation (Table 4).
- `rerun-ml-table4/`: re-execution of the 48 ML configurations with these hyperparameters, with the per-comment predictions and confidences (`predictions/ml/run_<fold>.csv`) and a comparison with the original confusion matrices (`ml/reference_comparison.csv`).
- `llm_prompting/`: prompting results per fold and aggregated over the folds and executions (Table 5).
- `plm/comment_data_results.csv`: PLM results with the selected hyperparameters.
- `llm_fine_tuning/comment_data_results copy.csv`: Llama-3 ft results in each fold (Table 6).
- `temporal-results/`: results of svm, RoBERTa, and Llama-3 ft in each execution of the temporal evaluation.
- `predictions/temporal-prediction/`: per-comment predictions of the temporal evaluation (`ml/`, `plm/`, `llm/`) and their comparison with the cross-validation on the 4,408 comments in the test sets of both settings: `temporal_vs_old_comparison.csv` (Table 8) and `temporal_vs_old_fold_*_common_pairs.csv` (fold-wise precision, recall, and F1 score, Table 9).

### `results_analysis/`
- `ensembled_models.xlsx`: per-comment predictions of the best configuration of the 12 models and of the 47 ensembles, and the ensemble results (RQ4).
- `complexity_analysis.md`: time complexity of training and inference of svm, RoBERTa, Llama-3 ft, and their ensemble, and the cost of the ensemble relative to each model.
- `issue_type_and_category_analysis.xlsx`: results of the 12 best models per issue type and problem category (RQ5).
- `create_venn_diagram.py`: overlap of the false positives and false negatives of svm, RoBERTa, and Llama-3 ft.
- `select_random_sample.py`: random sample of misclassified comments for the qualitative analysis (RQ6-RQ7).
- `compute_agreement_index.py`: inter-rater agreement (Cohen's Kappa and Krippendorff's Alpha) of the qualitative analysis.
- `compare_temporal_with_old_predictions.py`, `temporal_vs_old_fold_results_common_pairs.py`: comparison of the temporal evaluation with the cross-validation (Tables 8 and 9).
- `explain_findings.py`: analyses that explain the main findings (Sections 4.1-4.4): the words that distinguish solution comments, including the "Created attachment" text (`1a_*`), the recall per solution code (`1b_*`), class weights vs. Smote (`2_*`), the false positives of prompted Llama-3 on comments with technical content (`3_*`), and the overlap of the errors of the three best models (`4_*`). Outputs and their description in `generated/explain_findings/`.
- `plot_error_overlap_venn.py`: Venn diagrams of the true positives, false negatives, and false positives of svm, RoBERTa, and Llama-3 ft (Section 4.4). Outputs in `generated/error_overlap/`.
- `significance_tests.py`: paired cluster bootstrap test over issues (10,000 resamples) for the comparisons of Table 10, with Holm correction within each RQ and metric. `generated/significance_tests/results.csv` has all tests, with the differences, 95% confidence intervals, and raw and Holm-adjusted p-values; `claims.csv` maps each comparison to the claim in the paper. The `wilcoxon_*` columns are a secondary fold-level test that is not reported in the paper. The method is described in `generated/significance_tests/README.md`.
- `significance_table.py`: generates the rows of Table 10 (`generated/significance_tests/table.tex`) from `results.csv`.

### `generalizability/`
Transferability study on 10 Chromium and 10 GnuCash issues (RQ8), with leave-one-issue-out evaluation under three settings: project-specific, cross-project (trained on Mozilla data), and hybrid training.
- `dataset/`: the labeled Chromium and GnuCash comments, with and without embeddings.
- `mlm_experiments.py`: svm in the three settings.
- `plm_fine_tuning.py`: fine-tunes RoBERTa on the Mozilla data; `plm_experiments_mp.py`: RoBERTa in the three settings.
- `h_llm_fine_tuning_experiments.py`: fine-tunes Llama-3 on the Mozilla data; `llm_experiments.py`: Llama-3 ft in the three settings.
- `predictions/`, `results/`: predictions and results of the experiments.
- `utils.py`: shared helper functions.

## Installation

Prerequisites: Conda, or Python 3.10.4.

```bash
conda create -n solution_identification python=3.10.4
conda activate solution_identification
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

Alternatively, with a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

The prompting experiments require [Ollama](https://ollama.com/) with Llama-3, the Gpt embeddings require an OpenAI API key, and fine-tuning Llama-3 requires access to `meta-llama/Meta-Llama-3-8B` on Hugging Face.

## Running the Experiments

`solution_localization.sh` runs the pipeline:

```bash
chmod +x solution_localization.sh
./solution_localization.sh
```

The scripts in `sources/` and `generalizability/` are run from the repository root, except `f_llm_prompting_experiments.py` and `g_compute_metrics_for_prompting.py`, which use paths relative to `sources/` and take `development` or `test` as argument:

```bash
cd sources
python f_llm_prompting_experiments.py test
python g_compute_metrics_for_prompting.py test
```

The `__main__` block of each script selects the folds and configurations to run.

## Machine Information

We used CPUs for the ML models and GPUs for the PLMs and Llama-3. To reduce the training time of the cross-validation, we used multiple GPUs on the following machines:

1. Machine 1:
   - CPU: AMD EPYC 7532, Memory: 1.5 TB
   - GPU: NVIDIA H100, Memory: 95830 MB
2. Machine 2:
   - CPU: AMD EPYC 7543, Memory: 2.0 TB
   - GPU: NVIDIA A40, Memory: 46068 MB (x8)
3. Machine 3:
   - CPU: AMD EPYC 9354, Memory: 1.0 TB
   - GPU: NVIDIA A100, Memory: 40960 MB
