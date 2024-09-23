#!/bin/bash

# Run data preparation
python sources/data_preparation.py

# Run the MLM models
python sources/mlm_train.py

# Run the PLM models
python sources/plm_train.py

# Run the LLM prompting experiments
# =================================
# Run experiments with prompt set
python3 sources/f_llm_prompting_experiments.py development
python3 sources/g_compute_metrics_for_prompting development
# Run experiments with test set
python3 sources/f_llm_prompting_experiments.py test
python3 sources/g_compute_metrics_for_prompting test

# Run the LLM Fine-tuning experiments
python sources/llm_train.py
