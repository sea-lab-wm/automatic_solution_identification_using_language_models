This is the replication package for the paper "A Case Study on Automatically Identifying Solution-Related Content in Issue Report Discussions via Language Models".

# Installation Process

## Prerequisite: 
1. Conda or
2. Python 3.10.4 (Jump to Environment Setup:Step 4 if you have this version of python installed)

## Environment Setup

### Create Conda Environment:
1. conda create -n "test_env" python=3.10.4

### Activate Conda Environment
2. conda activate test_env

(Only Use this to deactivate this conda environment)
### Deactivate Conda Environment
3. conda deactivate

### Install python required package
4. pip install -r requirements.txt

# Files:

## Dataset:
### dataset_construction/comment_data 

This folder contains:
1. dataset_construction/comment_data/annotation_data_all_codes.json:    All Annotation Data
2. dataset_construction/comment_data/coded_issue_data.json:             All Issue Comment Data
3. dataset_construction/comment_data/example_actual_comment_data.csv:   Prompt Data
4. dataset_construction/comment_data/manual_comment_mapping.csv:        Manually Mapped Data

### dataset_construction/comment_data/folds

This folder contains all comment data with 3 embeddings(LLama, BERT, GPT). These data are splitted in 10 folds for cross validation. Each fold have development dataset, train dataset, test dataset, validation dataset.

### results

This folder contains results of MLMs(results/ml), PLMs(results/plm) & LLM(results/llm).

# Run Models (To Run All Models, Run from Step #5):

## Data Preparation
1. python sources/data_preparation.py

This will create dataset_construction/comment_data/folds folder. 

## Run MLMs
2. python sources/mlm_train.py

This will only run MLMs and save the results in results/ml folder.

## Run PLMs
3. python sources/plm_train.py

This will only run PLMs and save the results in results/plm folder.

## Run LLM
4. python sources/llm_train.py

This will only run LLMs and save the results in results/llm folder.

## Run All
5. chmod +x solution_localization.sh

Provile User Permission to The solution_localization.sh File.

6. ./solution_localization.sh

Run solution_localization.sh 