#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from drqa import retriever
import json
import sys
from utils import whitelist, is_year
import copy
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--option', type=str, default='tfidf')
parser.add_argument('--format', type=str, required=True)
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class(args.option)(tfidf_path=args.model)

if args.format == 'table_construction':
    with open('data/all_passages.json') as f:
        cache = set(json.load(f).keys())
    logger.info('Finished loading the passage keys')
    new_cache = {}

def table_linker(kv):
    k, v = kv
    assert isinstance(k, str) and isinstance(v, dict)

    new_table = copy.deepcopy(v)
    new_table['data'] = []
    new_table['header'] = [(_, []) for _ in v['header']]
    
    for row in v['data']:
        new_row = []
        for cell in row:
            if not whitelist(cell) or is_year(cell):
                new_row.append((cell, []))
                continue
            guessing = '/wiki/{}'.format(cell.replace(' ', '_'))
            if guessing in cache:
                new_row.append((cell, [guessing]))
                continue
            if cell in new_cache:
                new_row.append((cell, new_cache[cell]))
                continue

            try:
                doc_name, doc_scores = ranker.closest_docs(cell, 1)
                assert isinstance(doc_name, list)
                new_row.append((cell, doc_name))
                new_cache[cell] = doc_name
            except Exception:
                new_row.append((cell, []))

        assert len(new_row) == len(v['header'])
        new_table['data'].append(new_row)
    assert len(new_table['data']) == len(v['data'])
    return k, new_table

if __name__ == '__main__':
    if args.format == 'question_table':
        with open(f'released_data/{args.split}.json', 'r') as f:
            data = json.load(f)
        for k in [1, 5, 10, 20, 50]:
            succ = 0
            for i, d in enumerate(data):
                groundtruth_doc = d['table_id']
                query = d['question']
                doc_names, doc_scores = ranker.closest_docs(query, k)
                if groundtruth_doc in doc_names:
                    succ += 1
                sys.stdout.write('finished {}/{}; HITS@{} = {} \r'.format(i + 1, len(data), k, succ / (i + 1)))

            print('finished {}/{}; HITS@{} = {} \r'.format(i + 1, len(data), k, succ / (i + 1)))

    elif args.format == 'cell_text':
        with open('data/traindev_tables.json') as f:
            traindev_tables = json.load(f)
        with open('released_data/train_dev_test_table_ids.json') as f:
            tables_ids = set(json.load(f)['dev'])
        with open('link_generator/row_passage_query.json', 'r') as f:
            mapping = json.load(f)

        succ, prec_total, recall_total = 0, 0, 0
        for k, table in traindev_tables.items():
            if k not in tables_ids:
                continue

            for j, row in enumerate(table['data']):
                row_id = k + '_{}'.format(j)
                queries = mapping.get(row_id, [])
                gt_docs = []
                for cell in row:
                    gt_docs.extend(cell[1])
                doc_names = []
                for query in queries:
                    try:
                        doc_name, doc_scores = ranker.closest_docs(query, 1)
                        doc_names.extend(doc_name)
                    except Exception:
                        pass

                succ += len(set(gt_docs) & set(doc_names))
                prec_total += len(queries)
                recall_total += len(gt_docs)

                if len(queries) == 0 and len(gt_docs) > 0:
                    pass

            recall = succ / (recall_total + 0.01)
            precision = succ / (prec_total + 0.01)
            f1 = 2 * recall * precision / (precision + recall + 0.01)
            sys.stdout.write('F1@{} = {} \r'.format(1, f1))
        
        print('F1@{} = {}'.format(1, f1))

    elif args.format == 'table_construction':
        with open('data/all_plain_tables.json') as f:
            tables = json.load(f)
        logger.info('Finished loading the plain tables')
        
        n_threads = 64
        results = []
        with Pool(n_threads) as p:
            results = list(
                tqdm(
                    p.imap(table_linker, tables.items(), chunksize=16),
                    total=len(tables),
                    desc="process tables",
                )
            )

        linked_tables = dict(results)
        with open('data/all_constructed_tables.json', 'w') as f:
            json.dump(linked_tables, f)

    else:
        raise NotImplementedError()

