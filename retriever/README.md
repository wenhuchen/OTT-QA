# Document Retriever
This part is mainly from https://github.com/facebookresearch/DrQA. Thank their authors for making it public.


## Building the table index files/converting to db/building tf-idf index
```
python build_tfidf.py --build_option title_sectitle_schema --out_dir tfidf_title_sectitle_schema
```

This script generate the index representation of each table using title, section title and table schema.
