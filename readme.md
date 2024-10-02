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

# Directory Structure

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

1. ```a_annotation_and_issue_data_mapping.py```: This script map the annotation data with the issue data and assign each issue comment an annotation code according to annotation data.
2. ```b_preprocess_data.py```: This script preprocess the comment data to be used in MLMs experiments. 
3. ```c_generate_embeddings.py```: This script generate LM embeddings (Llama, GPT, BERT) to be used in MLMs experiments.
4. ```d_mlm_experiments.py```: This script run all MLMs and save the results to ```results/ml``` folder.
5. ```e_plm_experiments.py```: This script run all PLMs and save the results to ```results/plm``` folder.
6. ```f_llm_prompting_experiments.py```: This script run the LLM prompting experiments.
7. ```g_compute_metrics_for_prompting.py```: This script save the results to ```results/llm_prompting``` folder.
8. ```h_llm_fine_tuning_experiments.py```: This script run the LLM finetuning experiments and save results to ```results/llm_fine_tuning``` folder.

# Run Models:
1. chmod +x solution_localization.sh

Provile User Permission to The solution_localization.sh File.

2. ./solution_localization.sh

Run solution_localization.sh 

## Machine Information
To run MLMs we have used CPUs and for PLM and LLM we have used GPUs. Since We have used cross validation, to put less time on training models we have used multiple GPUs. Below we mentioned the machine information we have used to train and evaluate our models.

#### 1. ```Machine 1```: 
    1.1. CPU: AMD EPYC 7532, Memory: 1.5 TB
    1.2. GPU: NVIDIA H100, Memory: 95830 MB
#### 2. ```Machine 2```: 
    2.1. CPU: AMD EPYC 7543, Memory: 2.0 TB
    2.2. GPU: NVIDIA A40, Memory: 46068 MB (x8)
#### 3. ```Machine 3```: 
    3.1. CPU: AMD EPYC 9354, Memory: 1.0 TB
    3.2. GPU: NVIDIA A100, Memory: 40960 MB

