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
In this step, the model will extract tables and their hyperlinks, which is used to request the www to obtain the passages associated with it. Finally, the tokenized data of tables_tok/ and request_tok/ will be generated for the next step.

## Packed table and request files [Optional]
The pipeline.py will generate each table as a separate file, so there will be totally over 400K files in your system, which could potentially cause some issue. If you want to have a packed table and request file. Please download from [all_tables.json](https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_tables.json) and [all_request.json](https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_requests.json)
