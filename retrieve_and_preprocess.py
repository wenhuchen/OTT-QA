from preprocessing import *
from drqa import retriever
import json
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, type=str)
    parser.add_argument('--model', default=None, type=str)    
    args = parser.parse_args()

    pool = Pool(64)
    if args.split in ['dev', 'test']: 
        split = args.split
        with open(f'released_data/{split}.before_retrieval.json', 'r') as f:
            dev_data = json.load(f)
        k = 5       
        ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
        for i, d in enumerate(dev_data):
            query = d['question']
            doc_names, doc_scores = ranker.closest_docs(query, k)
            d['table_id'] = doc_names[0]
            d['top_k'] = doc_names
            d['top_k_scores'] = doc_scores.tolist()
        with open(f'preprocessed_data/{split}.json', 'w') as f:
            json.dump(dev_data, f, indent=2)

        results1 = pool.map(IR, dev_data)
        results2 = pool.map(CELL, results1)
        with open(f'preprocessed_data/{split}_linked.json', 'w') as f:
            json.dump(results2, f, indent=2)

        dev_inputs = generate_inputs(results2)
        with open(f'preprocessed_data/{split}_inputs.json', 'w') as f:
            json.dump(dev_inputs, f, indent=2)
    elif args.split == 'train':
        with open('released_data/train.json', 'r') as f:
            train_data = json.load(f)       

        results1 = pool.map(IR, train_data)
        results2 = pool.map(CELL, results1)
        train_results = analyze(results2)
        with open('preprocessed_data/train_linked.json', 'w') as f:
            json.dump(train_results, f, indent=2)
 
        results = prepare_stage1_data(train_results)
        with open('preprocessed_data/stage1_training_data.json', 'w') as f:
            json.dump(results, f, indent=2)

        results = pool.map(prepare_stage2_data, train_results)
        train_split = []
        for r1 in results:
            train_split.extend(r1)
        with open('preprocessed_data/stage2_training_data.json', 'w') as f:
            json.dump(train_split, f, indent=2)

        results = prepare_stage3_data(train_results)
        with open('preprocessed_data/stage3_training_data.json', 'w') as f:
            json.dump(results, f, indent=2)
    else:
        raise NotImplementedError

    pool.close()
    pool.join()
