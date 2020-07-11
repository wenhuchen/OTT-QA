# Document Retriever
This part is mainly from https://github.com/facebookresearch/DrQA. Thank their authors for making it public.


## Building the table indexes files
```
python build_corpus.py --build 1,2,4
```
This script generate the index representation of each table using title, section title and table schema.

## Storing the Documents
To create a sqlite db from a corpus of documents, run:
```
python build_db.py data/tf-idf-input.json data/WikiTables_124.db
```

## Building TF-IDF index
To build a TF-IDF index.
```
python build_tfidf.py data/WikiTables_124.db data/WikiTables_124.db tf-idf-124
```
The file will generate a indexing numpy file in the folder of tf-idf-124/
