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

with open('released_data/test.oracle_retrieval.json', 'r') as f:
    data = json.load(f)

if not args.debug:
    if not args.cell:
        if args.format == 'table':
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
        elif args.format == 'text':
            print("running ablation study to retrieve text directly")
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
    else:
        if args.format == 'text':
            print("running cell level retrieval for linked text")
            pairwise_info = []
            for d in data:
                table_id = d['table_id']
                with open('data/tables_tok/{}.json'.format(table_id), 'r') as f:
                    table = json.load(f)
                
                meta_info = cell_representation(table)
                for row in table['data']:
                    for cell in row:
                        for sub_cell, gt in zip(cell[0], cell[1]):
                            if sub_cell and len(sub_cell) > 1:
                                request = meta_info + [sub_cell]
                                pairwise_info.append((request, gt))
            with open('data/table_cell_retrieval.json', 'w') as f:
                json.dump(pairwise_info, f, indent=2)

            for k in [1, 5, 10, 20, 50]:
                succ = 0
                fail = 0
                for i, d in enumerate(pairwise_info):
                    query = use_what(d[0], args.usage)
                    try:
                        doc_names, doc_scores = ranker.closest_docs(query, k)
                        if d[1] and d[1] in doc_names:
                            succ += 1
                        elif d[1] and d[1] not in doc_names:
                            fail += 1
                        else:
                            pass
                    except Exception:
                        pass
                    sys.stdout.write('finished {}/{}; HITS@{} = {} \r'.format(i + 1, len(pairwise_info), k, succ / (succ + fail)))
                print('finished {}/{}; HITS@{} = {}'.format(i + 1, len(pairwise_info), k, succ / (succ + fail)))
        
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

else:
    for i, d in enumerate(data):
        groundtruth_doc = d['table_id']
        query = d['question']
        doc_names, doc_scores = ranker.closest_docs(query, 5)
        print(query)
        print(groundtruth_doc)
        print(doc_names)
        input("Hit key to see next one!")

