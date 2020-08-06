import copy
import json

def transform(original_table, debug=False):
  start_index = 0
  while start_index < len(original_table):
    if not original_table[start_index][0]['is_header']:
      start_index += 1
    else:
      break
  if start_index >= len(original_table):
    raise NotImplementedError()
  if debug:
    print("starting from {}th row".format(start_index))

  reserved = {}
  headers = []
  total_columns = 1000
  while start_index < len(original_table):
    row = copy.copy(original_table[start_index])
    tmp = []
    
    if len(headers) > 0 and len(reserved) == 0:
      break
    
    for j in range(0, total_columns):
      if j < len(tmp):
        continue

      if j in reserved:
        current_idx = len(tmp)
        tmp.append(reserved[current_idx][0])
        reserved[current_idx][1] -= 1
        if reserved[current_idx][1] == 0:
          del reserved[current_idx]
      else:
        if len(row) == 0 and len(headers) == 0:
          # The first row needs to determine the maximum column number
          total_columns = len(tmp)
          break
        else:
          cell = row.pop(0)
          if cell['is_header']:
            tmp.extend([cell['value']] * cell['column_span'])

            if cell['row_span'] > 1:
              reserved[j] = [cell['value'], cell['row_span'] - 1]
              if cell['column_span'] > 1:
                for k in range(1, cell['column_span']):
                  reserved[j + k] =  [cell['value'], cell['row_span'] - 1]

    if len(headers) == 0:  
      headers.extend(tmp)
    else:
      for i in range(len(headers)):
        headers[i] += ' - ' + tmp[i]

    start_index += 1
  if debug:
    print("Finished with headers: {}".format(headers))

  if start_index >= len(original_table):
    raise NotImplementedError()

  total_columns = len(headers)

  data = original_table[start_index:]
  rows = []
  reserved = {}
  for i, row in enumerate(data):
    row = copy.copy(row)
    tmp = []
    col_idx = 0
    for j in range(total_columns):
      if j < len(tmp):
        continue

      if j in reserved:
        current_idx = len(tmp)
        tmp.append(reserved[current_idx][0])
        reserved[current_idx][1] -= 1
        if reserved[current_idx][1] == 0:
          del reserved[current_idx]
      else:
        assert len(row) > 0
        cell = row.pop(0)
        tmp.extend([cell['value']] * cell['column_span'])

        if cell['row_span'] > 1:
          reserved[j] = [cell['value'], cell['row_span'] - 1]
          if cell['column_span'] > 1:
            for k in range(1, cell['column_span']):
              reserved[j + k] =  [cell['value'], cell['row_span'] - 1]
    
    assert len(row) == 0
    rows.append(tmp)

  return headers, rows
  
if __name__ == '__main__':
  error = 0
  output_dict = {}
  for i in range(len(pair)):
    try:
      header, data = transform(pair[i]['table'])
      table_idx = pair[i]['table_page_title'] + '_{}'.format(pair[i]['example_id'])
      output_dict[table_idx] = {'header': header, 'data': data, 'url': pair[i]['table_webpage_url'], 'title': pair[i]['table_page_title'], 'section_title': pair[i]['table_section_title'], 'section_text': pair[i]['table_section_text'], 'intro': ''}
    except Exception:
      error += 1
      #print("error with {}".format(i))
  print("failing conversion rate = {}".format(error / len(pair)))

  with open('data/totto_tables.json', 'w') as f:
    json.dump(output_dict, f, indent=2)
