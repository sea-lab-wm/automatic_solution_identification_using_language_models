#!/bin/bash

# Run data preparation
python sources/data_preparation.py

# Run the MLM models
python sources/mlm_train.py

# Run the PLM models
python sources/plm_train.py

# Run the LLM models
python sources/llm_train.py
