from preprocessing import *
from drqa import retriever
import json
import sys
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--model', default=None, type=str)    
    args = parser.parse_args()

    n_threads = 64
    if args.split in ['train', 'dev']:
        print("using {}".format(args.split))
        table_path = 'traindev_tables_tok'
        request_path = 'traindev_request_tok'
        if not os.path.exists(f'preprocessed_data/{args.split}_linked.json'):
            with open(f'released_data/{args.split}.traced.json', 'r') as f:
                data = json.load(f)

            results1 = []
            with Pool(n_threads) as p:
                func_ = partial(
                    IR,
                    table_path=table_path,
                    request_path=request_path
                )
                results1 = list(
                    tqdm(
                        p.imap(func_, data, chunksize=16),
                        total=len(data),
                        desc="convert examples to features",
                    )
                )

            results2 = []
            with Pool(n_threads) as p:
                func_ = partial(
                    CELL,
                    table_path=table_path,
                )
                results2 = list(
                    tqdm(
                        p.imap(func_, results1, chunksize=16),
                        total=len(results1),
                        desc="convert examples to features",
                    )
                )

            train_results = analyze(results2, table_path)
            random.shuffle(train_results)
            with open(f'preprocessed_data/{args.split}_linked.json', 'w') as f:
                json.dump(train_results, f, indent=2)

        if args.split == 'train':
            with open(f'preprocessed_data/{args.split}_linked.json', 'r') as f:
                train_results = json.load(f)

            results = prepare_stage1_data(train_results, table_path)
            with open('preprocessed_data/stage1_training_data.json', 'w') as f:
                json.dump(results, f, indent=2)


            results = []
            with Pool(n_threads) as p:
                func_ = partial(
                    prepare_stage2_data,
                    table_path=table_path,
                    request_path=request_path
                )
                results = list(
                    tqdm(
                        p.imap(func_, train_results, chunksize=16),
                        total=len(train_results),
                        desc="convert examples to features",
                    )
                )

            train_split = []
            for r1 in results:
                train_split.extend(r1)
            with open('preprocessed_data/stage2_training_data.json', 'w') as f:
                json.dump(train_split, f, indent=2)

            results = prepare_stage3_data(train_results, request_path)
            with open('preprocessed_data/stage3_training_data.json', 'w') as f:
                json.dump(results, f, indent=2)
            
    elif args.split in ['dev_retrieve', 'test_retrieve']: 
        split = args.split.split('_')[0]
        with open(f'released_data/{split}.json', 'r') as f:
            dev_data = json.load(f)
        k = 1
        with open('data/all_constructed_tables.json', 'r') as f:
            all_tables = json.load(f)
        with open('data/all_passages.json', 'r') as f:
            all_requests = json.load(f)

        print('Start Retrieving tables and requested documents')
        assert args.model is not None
        ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
        for d in dev_data:
            query = d['question']
            doc_names, doc_scores = ranker.closest_docs(query, k)
            d['table_id'] = doc_names[0]
            d['table'] = all_tables[d['table_id']]
            requested_documents = {}
            for row in d['table']['data']:
                for cell in row:
                    for ent in cell[1]:
                        requested_documents[ent] = all_requests[ent]
            d['requested_documents'] = requested_documents
        print('Done Retrieving tables and requested documents')

        results1 = []
        with Pool(n_threads) as p:
            results1 = list(
                tqdm(
                    p.imap(IR, dev_data, chunksize=32),
                    total=len(dev_data),
                    desc="convert examples to features",
                )
            )
        for d in results1:
            d['table'] = all_tables[d['table_id']]
        results2 = []
        with Pool(n_threads) as p:
            results2 = list(
                tqdm(
                    p.imap(CELL, results1, chunksize=32),
                    total=len(results1),
                    desc="convert examples to features",
                )
            )
        for d in results2:
            d['table'] = all_tables[d['table_id']]
        dev_inputs = generate_inputs(results2)
        with open(f'preprocessed_data/{split}_inputs.json', 'w') as f:
            json.dump(dev_inputs, f, indent=2)
    else:
        raise NotImplementedError
