# Crawling Wikipedia Tables

## Download the original HTML files from Wikipedia
```
	wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/htmls_containing_tables.zip
	wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/Wikipedia.zip
	unzip htmls_containing_tables.zip
	unzip Wikipedia.zip
```
After this step, you will have a htmls/ folder containing thousands of htmls files with plausible Wikipedia tables contained.

## Extracting tables and preprocess them
```
	python pipeline.py
```
In this step, the model will extract tables and their hyperlinks, which is used to request the www to obtain the passages associated with it. Finally, the tokenized data of tables_tok/ and request_tok/ will be generated to the local directory. The data/all_plain_tables.json and data/all_passages.json are generated to the parent data folder.
