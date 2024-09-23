This is the replication package for the paper "A Case Study on Automatically Identifying Solution-Related Content in Issue Report Discussions via Language Models".

# Installation Process

## Prerequisites
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

# Directorie Structure

## dataset
This folder contains all the dataset used in the paper. The folder contains following subfolders:
1. ```dataset/annotation_data```: This folder contains the annotation data, i.e., all the comments for all the 356 issues with the annotated code assigned by human annotators.
2. ```dataset/issue_data```: This folder contains the issue title, summary, and meta-data information for the 356 issues.
3. ```solution_identification_data```: This folder contains the solution identification data, i.e., the issue comments labeled as either solution or non-solution and dataset split (prompt set, train set, and test set).

## prompting
This folder contains 10 prompts templates used for the prompting experiments. It also contains the generated prompts and responses for all the 10 prompts with all the 10 folds test dataset for three runs.

## results
This folder contains results of MLMs, PLMs, LLM-prompting, and LLM-fine-tuning experiments.

## results_analysis
This folder contains the ensembled models analysis and the results analysis across issue types and problems categories.

## sources
This folder contains the source code for the data preparation, training, and evaluation of the models for all experiments. This folder contains the following files:
TODO: Mehedi, please update the list of files here.
1. ```a_annotation_and_issue_data_mapping.py```: This script will map the annotation data with the issue data and assign each issue comment an annotation code according to annotation data.
2. 

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