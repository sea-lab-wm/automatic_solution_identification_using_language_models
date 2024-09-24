#!/bin/bash

# Mapping Annotation and Issue Data
# =================================
# Run Mapping and Dataset Creation
python sources/a_annotation_and_issue_data_mapping.py

# Generate LM Embeddings
python sources/c_generate_embeddings.py

# Run the MLM models
python sources/d_mlm_experiments.py
# Run the PLM models
python sources/e_plm_experiments.py

# Run the LLM prompting experiments
# =================================
# Run experiments with prompt set
python sources/f_llm_prompting_experiments.py development
python sources/g_compute_metrics_for_prompting development
# Run experiments with test set
python sources/f_llm_prompting_experiments.py test
python sources/g_compute_metrics_for_prompting test

# Run the LLM Fine-tuning experiments
python sources/h_llm_fine_tuning_experiments.py
