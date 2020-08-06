"""TOTTO table processsing.

Processing the data format in TOTTO into the HybridQA one
"""

import copy
import hashlib
import json
import multiprocessing
from multiprocessing import Pool
import re

import nltk


def clean_cell_text(string):
  """Strip off the weird tokens."""
  string = string.replace('"', '')
  string = string.rstrip('^')
  string = string.replace('–', '-')
  string = string.replace('( ', '(')
  string = string.replace(' )', ')')
  string = string.replace('"', '')
  string = string.replace(u'\u00a0', u' ')
  string = string.replace('\n', ' ')
  string = string.rstrip('^')
  string = string.replace('\u200e', '')
  string = string.replace('\ufeff', '')
  string = string.replace(u'\u2009', u' ')
  string = string.replace(u'\u2010', u' - ')
  string = string.replace(u'\u2011', u' - ')
  string = string.replace(u'\u2012', u' - ')
  string = string.replace(u'\u2013', u' - ')
  string = string.replace(u'\u2014', u' - ')
  string = string.replace(u'\u2015', u' - ')
  string = string.replace(u'\u2018', u'')
  string = string.replace(u'\u2019', u'')
  string = string.replace(u'\u201c', u'')
  string = string.replace(u'\u201d', u'')
  string = re.sub(r' +', ' ', string)
  string = string.strip()
  string = nltk_tokenize(string)
  return string


def hash_string(string):
  sha = hashlib.sha256()
  sha.update(string.encode())
  return sha.hexdigest()[:16]


def nltk_tokenize(string):
  toks = nltk.word_tokenize(string)
  return ' '.join(toks)


def transform(original_table, debug=False):
  start_index = 0
  while start_index < len(original_table):
    if len(original_table[start_index]
          ) <= 1 or not original_table[start_index][0]['is_header']:
      start_index += 1
    else:
      break
  if start_index >= len(original_table):
    raise NotImplementedError()
  if debug:
    print('starting from {}th row'.format(start_index))

  reserved = {}
  headers = []
  total_columns = 1000
  while start_index < len(original_table):
    row = copy.copy(original_table[start_index])
    tmp = []

    if headers and (not reserved):
      break

    for j in range(0, total_columns):
      if j < len(tmp):
        continue

      if j in reserved:
        tmp.append(reserved[j][0])
        reserved[j][1] -= 1
        if reserved[j][1] == 0:
          del reserved[j]
      else:
        if (not row) and (not headers):
          # The first row needs to determine the maximum column number
          total_columns = len(tmp)
          break
        else:
          if not row:
            raise NotImplementedError()
          cell = row.pop(0)
          value = clean_cell_text(cell['value'])
          if cell['is_header']:
            tmp.extend([value] * cell['column_span'])

          if cell['row_span'] > 1 and cell['row_span'] < 10:
            reserved[j] = [value, cell['row_span'] - 1]
            if cell['column_span'] > 1:
              for k in range(1, cell['column_span']):
                reserved[j + k] = [value, cell['row_span'] - 1]

    if not headers:
      headers.extend(tmp)
    else:
      if len(headers) == len(tmp):
        for i in range(len(headers)):
          headers[i] += ' - ' + tmp[i]

    start_index += 1
  if debug:
    print('Finished with headers: {}'.format(headers))

  if start_index >= len(original_table):
    raise NotImplementedError()

  total_columns = len(headers)

  rows = []
  reserved = {}
  mapping = {}
  for i in range(start_index, len(original_table)):
    row = copy.copy(original_table[i])
    tmp = []
    counter = 0
    for j in range(total_columns):
      if j < len(tmp):
        continue

      if j in reserved:
        tmp.append(reserved[j][0])
        reserved[j][1] -= 1
        if reserved[j][1] == 0:
          del reserved[j]
      else:
        if not row:
          raise NotImplementedError()
        mapping[(i, counter)] = (len(rows), j)
        cell = row.pop(0)
        counter += 1
        value = clean_cell_text(cell['value'])
        tmp.extend([value] * cell['column_span'])

        if cell['row_span'] > 1 and cell['row_span'] < 10:
          reserved[j] = [value, cell['row_span'] - 1]
          if cell['column_span'] > 1:
            for k in range(1, cell['column_span']):
              reserved[j + k] = [value, cell['row_span'] - 1]

    if row:
      raise NotImplementedError()
    rows.append(tmp)

  return headers, rows, mapping


def get_table_sent(entry):
  try:
    header, data, mapping = transform(entry['table'])
    table_idx = entry['table_page_title'].replace(' ', '_') + '_{}'.format(
        entry['example_id'])
    table = (table_idx, {
        'header': header,
        'data': data,
        'url': entry['table_webpage_url'],
        'title': entry['table_page_title'],
        'section_title': entry['table_section_title'],
        'section_text': entry['table_section_text'],
        'intro': ''
    })

    positive_cell = []
    for cell in entry['highlighted_cells']:
      if tuple(cell) in mapping:
        index = mapping[tuple(cell)]
        positive_cell.append([data[index[0]][index[1]], index, '', 'table'])

    questions = []
    if positive_cell:
      for example in entry['sentence_annotations']:
        sentence = clean_cell_text(example['final_sentence'])
        hash_code = hash_string(sentence)
        questions.append({
            'table_id': table_idx,
            'question': sentence,
            'answer-text': 'none',
            'answer-node': positive_cell,
            'version': 'TOTTO',
            'question_id': hash_code,
            'where': 'table'
        })
    return table, questions

  except NotImplementedError:
    return None, None


if __name__ == '__main__':
  filepath = '/usr/local/google/home/wenhuchen/Documents/TOTTO/totto_data/totto_dev_data.jsonl'
  pair = []
  with open(filepath, 'r') as f:
    for line in f:
      pair.append(json.loads(line))
  filepath = '/usr/local/google/home/wenhuchen/Documents/TOTTO/totto_data/totto_train_data.jsonl'
  with open(filepath, 'r') as f:
    for line in f:
      pair.append(json.loads(line))
  print('Finish loading local data')

  output_dict = {}
  sentences = []

  cpu_cores = multiprocessing.cpu_count()
  print('using {} cores'.format(cpu_cores))
  pool = Pool(cpu_cores)

  results = pool.map(get_table_sent, pair)
  print('Finish the running')

  error = 0
  for r in results:
    if r[0] and r[1]:
      output_dict[r[0][0]] = r[0][1]
      sentences.extend(r[1])
    else:
      error += 1

  print('failing conversion rate = {}'.format(error / len(pair)))
  print('successful tables = {}'.format(len(output_dict)))

  with open(
      '/usr/local/google/home/wenhuchen/Documents/TOTTO/totto_data//tables.json',
      'w') as f:
    json.dump(output_dict, f, indent=2)
  with open(
      '/usr/local/google/home/wenhuchen/Documents/TOTTO/totto_data//train.json',
      'w') as f:
    json.dump(sentences, f, indent=2)
