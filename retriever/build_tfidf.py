#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging
import glob

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers

import sqlite3
import json
import importlib.util

from tqdm import tqdm
from drqa.retriever import utils


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Building corpus
# ------------------------------------------------------------------------------

def build_corpus(build_option, tmp_file):
    fw = open(tmp_file, 'w')
    posts = []
    with open('../data/all_plain_tables.json', 'r') as f:
        tables = json.load(f)
    with open('../data/all_passages.json', 'r') as f:
        passages = json.load(f)
    if build_option == 'title':
        for _, table in tables.items():
            title = table['title']
            content = "{}".format(title)
            fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    elif build_option == 'title_sectitle':
        for _, table in tables.items():
            title = table['title']
            section_title = table['section_title']
            content = "{} | {}".format(title, section_title)
            fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    elif build_option == 'title_sectitle_sectext':
        for _, table in tables.items():
            title = table['title']
            section_title = table['section_title']
            section_text = table['section_text']
            if section_text == '':
                content = "{} | {}".format(title, section_title)
            else:
                content = "{} | {} | {}".format(title, section_title, section_text)
            fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    elif build_option == 'title_sectitle_schema':
        for _, table in tables.items():
            title = table['title']
            section_title = table['section_title']
            headers = []
            for h in table['header']:
                headers.append(' '.join(h[0]))
            headers = ' '.join(headers)
            content = "{} | {} | {}".format(title, section_title, headers)
            fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    elif build_option == 'title_sectitle_content':
        for _, table in tables.items():
            title = table['title']
            section_title = table['section_title']
            contents = []
            for h in table['header']:
                contents.append(' '.join(h[0]))
            for rows in table['data']:
                for row in rows:
                    contents.append(' '.join(row[0]))
            contents = ' '.join(contents)
            content = "{} | {} | {}".format(title, section_title, contents)
            fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    elif build_option == 'text':
        for k, v in passages.items():
            fw.write(json.dumps({'id': k, 'text': v}) + '\n')
        fw.close()
    elif build_option == 'text_title':
        for k, v in passages.items():
            v = k.replace('/wiki/', '')
            v = v.replace('_', ' ')
            if k and v:
                fw.write(json.dumps({'id': k, 'text': v}) + '\n')
        fw.close()
    else:
        raise NotImplementedError
    fw.close()

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init_preprocess(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((doc['id'], doc['text']))
    return documents


def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        os.remove(save_path)
        #raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    workers = ProcessPool(num_workers, initializer=init_preprocess, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts, idf_cnts, option='tf-idf'):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    # Computing the IDF parameters
    Ns = get_doc_freqs(idf_cnts)
    idfs = np.log((idf_cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    if option == 'tfidf':
        # Computing the TF parameters
        tfs = cnts.log1p()
    elif option == 'bm25':
        k1 = 1.5
        b = 0.75
        # Computing the saturation parameters
        doc_length = np.array(cnts.sum(0)).squeeze()
        doc_length_ratio = k1 * (1 - b + b * doc_length / doc_length.mean())
        doc_length_ratio = sp.diags(doc_length_ratio, 0)
        binary = (cnts > 0).astype(int) 
        masked_length_ratio = binary.dot(doc_length_ratio)
        denom = cnts.copy()
        denom.data = denom.data + masked_length_ratio.data
        tfs = cnts * (1 + k1)
        tfs.data = tfs.data / denom.data
    else:
        raise NotImplementedError
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_option', type=str, default=None, 
                        help='Build option for corpus')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--option', type=str, default='tfidf',
                        help='TF-IDF or BM25')
    parser.add_argument('--tmp_file', type=str, default='/tmp/tf-idf-input.json', 
                        help='Tmp file to put build corpus')
    parser.add_argument('--tmp_db_file', type=str, default='/tmp/db.json', 
                        help='Tmp DB file to put build corpus')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    args = parser.parse_args()
    args.option = args.option.lower()
    assert args.option in ['tfidf', 'bm25'], "only support TF-iDF and BM25"

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logging.info('Building corpus...')
    build_corpus(args.build_option, args.tmp_file)

    logging.info('Building DB file...')
    store_contents(
        args.tmp_file, args.tmp_db_file, args.preprocess, args.num_workers)

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.tmp_db_file}
    )
    idf_count_matrix, _ = get_count_matrix(
        args, 'sqlite', {'db_path': args.tmp_db_file}
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix, idf_count_matrix, option=args.option)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(idf_count_matrix)

    basename = 'index'
    basename += ('-%s-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.option, args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    retriever.utils.save_sparse_csr(filename, tfidf, metadata)
