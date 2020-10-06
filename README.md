# OpenDomain Hybrid Question Answering

This respository contains the code and data for open-domain hybrid question answering for both tabular and textual data. In the previous hybridQA setting, the groundtruth tables and passages are already provided, the model only needs to perform multi-hop reasoning to acquire the answer. In contrast, this project proposes a more challenging setting where the model needs to retrieve tables from the whole Wikipedia to find the one for answering the given questions.

What's new compared to [HybridQA](http://hybridqa.github.io/):
- The questions are re-annotated, which means that the questions are de-contextualized to be standalone without relying on the given context to understand.
- The questions are collected with strict mannual quality control, they are of high quality.
- We already crawled all the 'hard negative' tables from Wikipedia and format them in the same manner as the ground truth tables.
- We pair the tables with their surrounding information like Wikipedia title, section title and section text. 


## Folder Hierarchy
- released_data: this folder contains the question/answer pairs for training, dev and test data.
- data/traindev_tables_tok: this folder contains the 420K+ preprocessed pool of tables.
- data/traindev_request_tok: this folder cotains their associated 2M passages from Wikipedia.
- table_crawling/: the folder contains the information about how we extract tables from Wikipedia.
- retriever/: the folder contains the information about how we build index for the table and text.
- data/all_plain_tables.json: this file contains the 400K+ open-domain tables for the test set. (see the following downloading command)
- data/all_passages.json: this file contains the 1M+ open-domain passages for the test set. (see the following downloading command)

## Requirements
- [HuggingFace](https://github.com/huggingface/transformers)
- [DocQA](https://github.com/facebookresearch/DrQA)
- [Pytorch 1.4](https://pytorch.org/)

We suggest using virtual environment to install these dependencies.
```
pip install transformers

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

git clone https://github.com/facebookresearch/DrQA.git
cd DrQA; pip install -r requirements.txt; python setup.py develop
```

## Additional Information
If you want to know more about the crawling procedure, please refer to [crawling](https://github.com/wenhuchen/OpenHybridQA/tree/master/table_crawling) for details.

If you want to know more about the retrieval procedure, please refer to [retriever](https://github.com/wenhuchen/OpenDomainHybridQA/tree/master/retriever) for details.

Or you can skip these two steps to directly download the needed files from AWS in Step1.

## Step0: Run retrieval experimens:
```
python retriever_baseline.py --model retriever/tfidf_title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz
```
This script will print the recall@1,5,10,20,50 for TF-IDF retrieval under title+title section+schema setting.

## Step1: Download the necessary files 
```
unzip tf-idf-index.zip
cd data/
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../
```
This command will download the crawled tables and linked passages from Wikiepdia in a cleaned format.

## Step2: Preprocessing the train/dev/test files
```
python retrieve_and_preprocess.py --split train
python retrieve_and_preprocess.py --split dev --model retriever/tfidf_title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz
```
This command will generate training data for different submodules in the following steps. For dev/test, it will also retrieve tables from the pool.

## Step3: Train the three modules in the reader.
```
python train_stage12.py --do_lower_case --do_train --train_file preprocessed_data/stage1_training_data.json --learning_rate 2e-6 --option stage1 --num_train_epochs 3.0 --model_name_or_path bert-large-uncased
python train_stage12.py --do_lower_case --do_train --train_file preprocessed_data/stage2_training_data.json --learning_rate 5e-6 --option stage2 --num_train_epochs 3.0 --model_name_or_path bert-large-uncased
python train_stage3.py --do_train  --do_lower_case   --train_file preprocessed_data/stage3_training_data.json  --per_gpu_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 4.0   --max_seq_length 384   --doc_stride 128  --threads 8 --model_name_or_path bert-large-uncased
```
The three commands separately train the step1, step2 and step3 neural modules, all of them are based on BERT-uncased-base model from HugginFace implementation.

## Step4: Evaluation
```
python train_stage12.py --stage1_model stage1/[YOUR-MODEL-FOLDER] --stage2_model stage2/[YOUR-MODEL-FOLDER] --do_lower_case --predict_file preprocessed_data/dev_inputs.json --do_eval --option stage12
python train_stage3.py --model_name_or_path stage3/[YOUR-MODEL-FOLDER] --do_stage3   --do_lower_case  --predict_file predictions.intermediate.json --per_gpu_train_batch_size 12  --max_seq_length 384   --doc_stride 128 --threads 8
```
Once you have generated the predictions.json file, you can use the following command to see the results.
```
python evaluate_script.py predictions.json released_data/dev_reference.json
```

