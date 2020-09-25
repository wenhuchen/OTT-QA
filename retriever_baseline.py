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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--option', type=str, default='tfidf')
parser.add_argument('--format', type=str, default='table')
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--cell', action='store_true', default=False)
parser.add_argument('--usage', type=str, default='content')
parser.add_argument('--offline_cell_classification', type=str, default=None)
parser.add_argument('--all_request', type=str, default=None)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class(args.option)(tfidf_path=args.model)

def cell_representation(table):
    title = table['title']
    section_title = table['section_title']
    return [title, section_title]

def use_what(whole_representation, usage):
    if usage == 'title+sectitle+content':
        return ' '.join(whole_representation[:3])
    elif usage == 'content':
        return whole_representation[2]
    elif usage == 'title+content':
        return whole_representation[0] + ' ' + whole_representation[2]
    elif usage == 'sectitle+content':
        return whole_representation[1] + ' ' + whole_representation[2]
    else:
        raise NotImplementedError()


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

elif args.format == 'question_text':
    with open(f'released_data/{args.split}.json', 'r') as f:
        data = json.load(f)
    for k in [1, 5, 10, 20, 50]:
        succ = 0
        fail = 0
        for i, d in enumerate(data):
            if d['where'] == 'passage':
                groundtruth_doc = []
                for node in d['answer-node']:
                    groundtruth_doc.append(node[2])

                query = d['question']
                doc_names, doc_scores = ranker.closest_docs(query, k)
                if any([_ in doc_names for _ in groundtruth_doc]):
                    succ += 1
                else:
                    fail += 1
                sys.stdout.write('finished {}/{}; HITS@{} = {} \r'.format(i + 1, len(data), k, succ / (succ + fail)))

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
                #doc_names.append('/wiki/' + query.replace(' ', '_'))
                try:
                    doc_name, doc_scores = ranker.closest_docs(query, 1)
                    doc_names.extend(doc_name)
                except Exception:
                    pass

            succ += len(set(gt_docs) & set(doc_names))
            prec_total += len(queries)
            recall_total += len(gt_docs)

            if len(queries) == 0 and len(gt_docs) > 0:
                #import pdb
                #pdb.set_trace()
                pass

        recall = succ / (recall_total + 0.01)
        precision = succ / (prec_total + 0.01)
        f1 = 2 * recall * precision / (precision + recall + 0.01)
        sys.stdout.write('F1@{} = {} \r'.format(1, f1))
    
    print('F1@{} = {}'.format(1, f1))

elif args.format == 'table_construction':
    with open('preprocessed_data/test.json', 'r') as f:
        required_test_tables = json.load(f)
    with open(args.offline_cell_classification, 'r') as f:
        offline_cell_classification = json.load(f)
    with open(args.all_request, 'r') as f:
        all_request = json.load(f)

    for entry in required_test_tables:
        table_id = entry['table_id']
        with open('data/plain_tables_tok/{}.json'.format(table_id), 'r') as f:
            table = json.load(f)

        for i, cell in enumerate(table['header']):
            table['header'][i] = ([cell], [None])

        requests = {}
        for i, row in enumerate(table['data']):
            for j, cell in enumerate(row):
                success = False
                if offline_cell_classification['{}_{}_{}'.format(table_id, i, j)] == 1:
                    query = use_what((None, None, cell), args.usage)
                    try:
                        doc_names, doc_scores = ranker.closest_docs(query, 1)
                        table['data'][i][j] = ([cell], [doc_names[0]])
                        requests[doc_names[0]] = all_request[doc_names[0]]
                        success = True
                    except Exception:
                        pass
                
                if not success:
                    table['data'][i][j] = ([cell], [None])
        
        with open('data/reconstructed_tables/{}.json'.format(table_id), 'w') as f:
            json.dump(table, f, indent=2)
        with open('data/reconstructed_request/{}.json'.format(table_id), 'w') as f:
            json.dump(requests, f, indent=2)
else:
    raise NotImplementedError()

