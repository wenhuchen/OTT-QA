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
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class(args.option)(tfidf_path=args.model)

with open('released_data/test.oracle_retrieval.json', 'r') as f:
    data = json.load(f)


if not args.debug:
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
    for i, d in enumerate(data):
        groundtruth_doc = d['table_id']
        query = d['question']
        doc_names, doc_scores = ranker.closest_docs(query, 5)
        print(query)
        print(groundtruth_doc)
        print(doc_names)
        input("Hit key to see next one!")

