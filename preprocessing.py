import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import nltk.data
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from utils import *
import re
import random
import nltk
import dateparser
from dateparser.search import search_dates
from dateparser import parse
import json
from multiprocessing import Pool
import os

stopWords = set(stopwords.words('english'))
tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
resource_path = 'data/'
best_threshold = 0.80

# Finding the longest substring
def longestSubstringFinder(S,T):
    S = S.lower()
    T = T.lower()
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])
    
    return longest, lcs_set

# Measure the longest string overlap distance
def longest_match_distance(str1s, str2s):
    longest_string = []
    for str1 in str1s:
        longest_string.append([])
        for str2 in str2s:
            length, _ = longestSubstringFinder(str1, str2)
            longest_string[-1].append(1 - length / len(str1))
    return longest_string


def IR(data_entry, table_path='traindev_tables_tok', request_path='traindev_request_tok'):
    table_id = data_entry['table_id']
    threshold = 0.99
    # Loading the table/request information
    with open(f'{resource_path}/{table_path}/{table_id}.json') as f:
        table = json.load(f)
    with open(f'{resource_path}/{request_path}/{table_id}.json') as f:
        requested_documents = json.load(f)

    # Mapping entity link to cell, entity link to surface word
    mapping_entity = {}
    for row_idx, row in enumerate(table['data']):
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                if ent:
                    if ent not in mapping_entity:
                        mapping_entity[ent] = {cell[0][i]: [(row_idx, col_idx)]}
                    else:
                        if cell[0][i] not in mapping_entity[ent]:
                            mapping_entity[ent][cell[0][i]] = [(row_idx, col_idx)]
                        else:
                            mapping_entity[ent][cell[0][i]] = mapping_entity[ent][cell[0][i]] + [(row_idx, col_idx)]
    
    # Convert the paragraph and question into TF-IDF features
    keys = []
    paras = []
    for k in mapping_entity:
        v = requested_documents[k]
        for _ in tokenizer.tokenize(v):
            keys.append(k)
            paras.append(_)
    
    qs = [data_entry['question']]

    # Find out the best matched passages based on distance
    tfidf_nodes = []    
    try:
        para_feature = tfidf.fit_transform(paras)
        transformed = True
    except Exception:
        print("only containing stop words, skip it")
        transformed = False
    
    if transformed:
        q_feature = tfidf.transform(qs)
        para_tfidf_dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]
        min_dist = {}
        tfidf_best_match = ('N/A', None, 1.)
        for k, para, d in zip(keys, paras, para_tfidf_dist):
            if d < min_dist.get(k, threshold):
                min_dist[k] = d
                if d < tfidf_best_match[-1]:
                    tfidf_best_match = (k, para, d)
                if d <= best_threshold:
                    for content, locs in mapping_entity[k].items():
                        for loc in locs:
                            tfidf_nodes.append((content, loc, k, para, d))
        
        if tfidf_best_match[0] != 'N/A':
            if tfidf_best_match[-1] > best_threshold:
                for content, locs in mapping_entity[k].items():
                    for loc in locs:
                        tfidf_nodes.append((content, loc, k, tfidf_best_match[1], tfidf_best_match[2]))

    # Find the best matched paragraph string
    string_nodes = []
    para_longest_string_match_dist = longest_match_distance(qs, paras)[0]
    min_dist = {}
    string_best_match = ('N/A', None, 1.)
    for k, para, d in zip(keys, paras, para_longest_string_match_dist):
        if d < min_dist.get(k, threshold):
            min_dist[k] = d
            if d < string_best_match[-1]:
                string_best_match = (k, para, d)
            if d <= best_threshold:
                for content, locs in mapping_entity[k].items():
                    for loc in locs:
                        string_nodes.append((content, loc, k, para, d))
                
    if string_best_match[0] != 'N/A':
        if string_best_match[-1] > best_threshold:
            for content, locs in mapping_entity[k].items():
                for loc in locs:
                    string_nodes.append((content, loc, k, string_best_match[1], string_best_match[2]))
    
    data_entry['tf-idf'] = tfidf_nodes
    data_entry['string-overlap'] = string_nodes

    return data_entry


def convert2num(string):
    string = string.replace(',', '')
    if string.endswith('%'):
        string = string.rstrip('%')
    try:
        string = float(string)
        return string
    except Exception:
        return None
    
def find_superlative(table_id, table):
    if not os.path.exists('{}/tmp/{}.json'.format(resource_path, table_id)):
        mapping = {}
        headers = [_[0][0] for _ in table['header']]
        for j in range(len(table['header'])):
            mapping[headers[j]] = []
            activate_date_or_num = None
            if headers[j] not in ['#', 'Type', 'Name', 'Location', 'Position', 'Category', 'Nationality',
                                  'School', 'Notes', 'Notability', 'Country']:
                for i, row in enumerate(table['data']):
                    if len(table['data'][i][j][0]) > 1:
                        continue

                    data = table['data'][i][j][0][0]
                    if data in ['', '-']:
                        continue

                    num = convert2num(data)
                    if num and data.isdigit() and num > 1000 and num < 2020 and activate_date_or_num in ['date', None]:
                        date_format = parse(data)
                        mapping[headers[j]].append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                        activate_date_or_num = 'date'
                    elif num and activate_date_or_num in ['num', None]: 
                        mapping[headers[j]].append((num, 'number', [data, (i, j), None, None, 1.0]))
                        activate_date_or_num = 'num'
                    else:
                        try:
                            date_format = parse(data)
                            if date_format and activate_date_or_num in ['date', None]:
                                mapping[headers[j]].append((date_format, 'date', [data, (i, j), None, None, 1.0]))
                                activate_date_or_num = 'date'
                        except Exception:
                            continue

            if len(mapping[headers[j]]) < 0.3 * len(table['data']):
                mapping[headers[j]] = []

        nodes = []
        for k, v in mapping.items():
            if len(v) > 0:
                tmp = sorted(v, key = lambda x: x[0])
                if tmp[0][1] == 'number':
                    tmp_node = tmp[0][-1]
                    tmp_node[3] = 'minimum'
                    nodes.append(tmp_node)
                    tmp_node = tmp[-1][-1]
                    tmp_node[3] = 'maximum'
                    nodes.append(tmp_node)
                else:
                    tmp_node = tmp[0][-1]
                    tmp_node[3] = 'earliest'
                    nodes.append(tmp_node)
                    tmp_node = tmp[-1][-1]
                    tmp_node[3] = 'latest'
                    nodes.append(tmp_node)

        with open('{}/tmp/{}.json'.format(resource_path, table_id), 'w') as f:
            json.dump(nodes, f)
    else:
        with open('{}/tmp/{}.json'.format(resource_path, table_id), 'r') as f:
            nodes = json.load(f)

    return nodes

def CELL(d, table_path='traindev_tables_tok'):
    threshold = 90
    # LINKING THE CELL DATA
    triggers = ['JJR', 'JJS', 'RBR', 'RBS']
    
    new_processed = []

    table_id = d['table_id']
    with open(f'{resource_path}/{table_path}/{table_id}.json') as f:
        table = json.load(f)        

    tmp_link = []
    for row_idx, row in enumerate(table['data']):
        for col_idx, cell in enumerate(row):
            if cell[0] != ['']:
                for ent in cell[0]:
                    ratio = fuzz.partial_ratio(' ' + ent.lower() + ' ', ' ' + d['question'].lower() + ' ')
                    if ratio > threshold:
                        tmp_link.append((ent, (row_idx, col_idx), None, 'string match', ratio / 100))

    d['links'] = tmp_link
    if any([_ in d['question_postag'] for _ in triggers]):
        try:
            tmp = find_superlative(table_id, table)
            d['links'] = d['links'] + tmp
        except Exception:
            print("failed with table {}".format(table_id))

    return d

def analyze(processed, table_path='traindev_tables_tok'):
    trivial, easy, medium, hard, no_answer, number, yesorno, repeated = 0, 0, 0, 0, 0, 0, 0, 0
    from_passage, from_cell, from_both = 0, 0, 0
    new_processed = []
    used_question_id = set([])
    
    question_type = ''
    for p in processed:
        if p['question_id'] in used_question_id:
            repeated +=1
            continue
        elif p['answer-text'].lower() in ['yes', 'no']:
            yesorno += 1
            continue
        elif len(p['answer-text'].split(' ')) > 15:
            yesorno += 1
            continue
        else:
            if len(p['answer-node']) > 1 and p['answer-node'][0][-1] == 'table':
                table_id = p['table_id']
                with open(f'{resource_path}/{table_path}/{table_id}.json', 'r') as f:
                    table = json.load(f)
                headers = [" , ".join(cell[0]) for cell in table['header']]
                potential_headers = set()
                for h in headers:
                    if " " + h.lower() + " " in " " + p['question'].lower() + " ":
                        potential_headers.add(h)

                if len(potential_headers) > 0:
                    remaining_nodes = []
                    for n in p['answer-node']:
                        if headers[n[1][1]] in potential_headers:
                            remaining_nodes.append(n)

                    if len(remaining_nodes) > 0 and len(remaining_nodes) < len(p['answer-node']):
                        p['answer-node'] = remaining_nodes

            # Categorize the difficulty level
            number_trigger = ['how many', 'how much', 'how long', 'how far', 'how old', 'difference', 'total']
            answer_node = p['answer-node']
            if len(answer_node) == 0:
                if any([x in p['question'].lower() for x in number_trigger]):
                    number += 1
                    question_type = 'numeric'
                else:
                    no_answer += 1
                    continue
            else:
                matching_cells = []
                if p['tf-idf']:
                    matching_cells.extend([tuple(_[1]) for _ in p['tf-idf']])
                if p['string-overlap']:
                    matching_cells.extend([tuple(_[1]) for _ in p['string-overlap']])
                linking_cells = [tuple(_[1]) for _ in p['links']]
                matching_cells = set(matching_cells)
                linking_cells = set(linking_cells)

                evidence_cells = matching_cells | linking_cells
                answer_row = set([_[1][0] for _ in answer_node])
                evidence_row = set([_[0] for _ in evidence_cells])
                intersect_row = answer_row & evidence_row

                if len(intersect_row) > 0:
                    new_answer_nodes = []
                    for node in p['answer-node']:
                        if node[1][0] in intersect_row:
                            new_answer_nodes.append(node)
                    p['answer-node'] = new_answer_nodes
                
                answer_cells = set([tuple(_[1]) for _ in p['answer-node']])
                if len(evidence_cells & answer_cells) > 0:
                    new_answer_nodes = []
                    for node in p['answer-node']:
                        if tuple(node[1]) in evidence_cells:
                            new_answer_nodes.append(node)
                    p['answer-node'] = new_answer_nodes
                
                answer_cells = set([tuple(_[1]) for _ in p['answer-node']])
                
                if matching_cells == answer_cells:
                    trivial += 1
                    question_type = 'trivial'
                elif len(evidence_cells & answer_cells) > 0:
                    easy += 1
                    question_type = 'easy'
                else:
                    if len(intersect_row) > 0:
                        medium += 1
                        question_type = 'medium'
                    else:
                        hard += 1
                        question_type = 'hard'

        p['type'] = question_type

        if len(answer_node) > 0:
            possible_passage, possibel_table = 0, 0
            for answer in answer_node:
                if answer[-1] == 'passage':
                    possible_passage += 1
                else:
                    possibel_table += 1
            # Trace back where it comes from
            if possible_passage > 0 and possibel_table > 0:
                p['where'] = 'both'
                from_both += 1
            elif possible_passage > 0:
                p['where'] = 'passage'
                from_passage += 1
            else:
                p['where'] = 'table'
                from_cell += 1
            p['where'] = answer_node[0][-1]
        else:
            raise ValueError('wrong parsing')
            
        new_processed.append(p)
        used_question_id.add(p['question_id'])

    print("trivial: {}, easy: {}, medium: {}, hard: {}, number: {}, no answer: {}, yes/no: {}, repeated: {}".
          format(trivial, easy, medium, hard, number, no_answer, yesorno, repeated))
    print("from cell: {}, from passage: {}, from both: {}".format(from_cell, from_passage, from_both))

    return new_processed

def generate_inputs(data, table_path='traindev_tables_tok'):
    split = []
    for d in data:
        table_id = d['table_id']
        with open(f'{resource_path}/{table_path}/{table_id}.json', 'r') as f:
            table = json.load(f)
        headers = [cell[0][0] for cell in table['header']]

        tmp = []
        labels = []
        for node in d['tf-idf']:
            tmp.append(list(node) + [headers[node[1][1]], 'tf-idf'])
        for node in d['string-overlap']:
            tmp.append(list(node) + [headers[node[1][1]], 'string-overlap'])            
        for node in d['links']:
            tmp.append(list(node) + [headers[node[1][1]], 'links'])
        split.append({'question': d['question'], 'question_id': d['question_id'], 
                      'table_id': d['table_id'], 'nodes': tmp})
    return split

def prepare_stage1_data(data, table_path='traindev_tables_tok'):
    split = []
    for d in data:
        if d['type'] in ['medium', 'easy']:
            table_id = d['table_id']
            with open(f'{resource_path}/{table_path}/{table_id}.json', 'r') as f:
                table = json.load(f)
            headers = [" , ".join(cell[0]) for cell in table['header']]

            answer_nodes = d['answer-node']
            answer_rows = set([_[1][0] for _ in answer_nodes])

            tmp = []
            labels = []
            for node in d['tf-idf']:
                tmp.append(list(node) + [headers[node[1][1]], 'tf-idf'])        
                if node[1][0] in answer_rows:
                    labels.append(1)
                else:
                    labels.append(0)

            for node in d['string-overlap']:
                tmp.append(list(node) + [headers[node[1][1]], 'string-overlap'])        
                if node[1][0] in answer_rows:
                    labels.append(1)
                else:
                    labels.append(0)

            for node in d['links']:
                tmp.append(list(node) + [headers[node[1][1]], 'links'])   
                if node[1][0] in answer_rows:
                    labels.append(1)
                else:
                    labels.append(0)
            
            split.append({'question': d['question'], 'question_id': d['question_id'], 'table_id': d['table_id'], 
                              'nodes': tmp, 'labels': labels})
    
    return split

def prepare_stage2_data(d, table_path='traindev_tables_tok', request_path='traindev_request_tok'):
    split = []
    if d['type'] in ['medium', 'easy']:
        table_id = d['table_id']
        with open(f'{resource_path}/{table_path}/{table_id}.json', 'r') as f:
            table = json.load(f)
        
        with open(f'{resource_path}/{request_path}/{table_id}.json', 'r') as f:
            requested_document = json.load(f)
        
        headers = [cell[0][0] for cell in table['header']]
        
        answer_nodes = d['answer-node']
        answer_rows = {_[1][0]: _ for _ in answer_nodes}

        labels = []
        for name, source in zip(['tf-idf', 'string-overlap', 'links'], [d['tf-idf'], d['string-overlap'], d['links']]):
            for node in source:
                i = node[1][0]
                if i in answer_rows and i >= 0:
                    tmp = {'question': d['question'], 'question_id': d['question_id'], 'table_id': d['table_id'], 
                            'current': list(node) + [headers[node[1][1]], name]}
                    target_nodes = []
                    labels = []
                    same_row = table['data'][i]
                    for j, cell in enumerate(same_row):
                        for content, url in zip(cell[0], cell[1]):
                            if len(content) > 0:
                                if url:
                                    doc = requested_document[url]
                                    intro = filter_firstKsents(doc, 1)
                                    target_nodes.append((content, (i, j), url, headers[j], intro))
                                    if url == answer_rows[i][2]:
                                        labels.append(1)
                                    else:
                                        labels.append(0)
                                else:
                                    target_nodes.append((content, (i, j), None, headers[j], ''))
                                    if content == answer_rows[i][0]:
                                        labels.append(1)
                                    else:
                                        labels.append(0)
                                
                        if len(cell[0]) > 1:
                            content = ' , '.join(cell[0])
                            if content == answer_rows[i][0]:
                                labels.append(1)
                            else:
                                labels.append(0)
                                
                            target_nodes.append((content, (i, j), None, headers[j], ''))
                        
                    tmp['labels'] = labels

                    assert sum(labels) > 0, d['question_id']
                    
                    tmp['target'] = target_nodes

                    split.append(tmp)

    return split

def prepare_stage3_data(data, request_path='traindev_request_tok'):
    split = []
    for d in data:
        if d['where'] == 'passage':
            table_id = d['table_id']
            
            with open(f'{resource_path}/{request_path}/{table_id}.json') as f:
                requested_documents = json.load(f)        
            
            #tmp = mapping.get(str(table_id), [])
            
            used = set()
            for node in d['answer-node']:
                if node[2] not in used:
                    context = requested_documents[node[2]]
                    context = 'Title : {} . '.format(node[0]) + context
                    
                    orig_answer = d['answer-text']

                    start = context.lower().find(orig_answer.lower())

                    if start == -1:
                        import pdb
                        pdb.set_trace()

                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1

                    answer = context[start:start+len(orig_answer)]
                    #assert(answer.lower() == orig_answer.lower(), "{} -> {}".format(answer, orig_answer))
                    
                    split.append({'context': context, 'title': table_id, 
                                      'question': d['question'], 'question_id': d['question_id'],
                                      'answers': [{'answer_start': start, 'text': answer}]})
                    used.add(node[2])
                else:
                    continue
        
        if d['where'] == 'table':
            table_id = d['table_id']
            
            with open(f'{resource_path}/{request_path}/{table_id}.json') as f:
                requested_documents = json.load(f)  
                
            used = set()
            for node in d['answer-node']:
                if node[2] and node[2] not in used:
                    context = requested_documents[node[2]]
                    context = 'Title : {} . '.format(node[0]) + context
                    
                    orig_answer = node[0]

                    start = context.lower().find(orig_answer.lower())

                    if start == -1:
                        import pdb
                        pdb.set_trace()

                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1
                        
                    answer = context[start:start+len(orig_answer)]
                    
                    split.append({'context': context, 'title': table_id, 
                                      'question': d['question'], 'question_id': d['question_id'],
                                      'answers': [{'answer_start': start, 'text': answer}]})
                    used.add(node[2])
                else:
                    continue
    return split
