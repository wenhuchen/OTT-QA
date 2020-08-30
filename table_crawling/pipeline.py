import time
import os
import sys
import json
import re
from bs4 import BeautifulSoup
from multiprocessing import Pool
import multiprocessing
import urllib3
import copy
from shutil import copyfile
from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.parse
import glob
import sys
import os

# HTTP manager
http = urllib3.PoolManager()
urllib3.disable_warnings()
sys.path.append("../")
from utils import *

# Initializing the resource folder
output_folder = 'data/'
input_htmls = 'htmls'
#default setting
#default_setting = {'miniumu_row': 8, 'ratio': 0.7, 'max_header': 6, 'min_header': 2}
default_setting = {'miniumu_row': 5, 'ratio': 0.85, 'max_header': 10, 'min_header': 2}

if len(sys.argv) == 2:
    steps = sys.argv[1].split(',')
else:
    steps = ['1', '2', '3', '4', '5', '6', '7']
print("performing steps = {}".format(steps))

if '3' in steps:
    with open('Wikipedia/wiki-intro-with-ents-dict.json', 'r') as f:
        cache = json.load(f)
    with open('Wikipedia/redirect_link.json', 'r') as f:
        redirect = json.load(f)
    with open('Wikipedia/old_merged_unquote.json', 'r') as f:
        dictionary = json.load(f)

def tokenize_merged(k_v):
    k, v = k_v
    return k, tokenize(v)

def process_link(text):
    tmp = []
    hrefs = []
    for t in text.find_all('a'):
        if len(t.get_text().strip()) > 0:
            if 'href' in t.attrs and t['href'].startswith('/wiki/'):
                tmp.append(t.get_text(separator=" ").strip())
                hrefs.append(t['href'].split('#')[0])
            else:
                tmp.append(t.get_text(separator=" ").strip())
                hrefs.append(None)
    if len(tmp) == 0:
        return [''], [None]
    else:
        return tmp, hrefs

def remove_ref(text):
    for x in text.find_all('sup'):
        x.extract()
    return text

def get_section_title(r):
    text = r.previous_sibling
    title_hierarchy = []
    while text is None or text == '\n' or text.name not in ['h2', 'h3']:
        if text is None:
            break
        else:
            text = text.previous_sibling               
    
    if text is not None:
        title_hierarchy.append(text.find(class_='mw-headline').text)
        if text.name in ['h3']:
            while text is None or text == '\n' or text.name not in ['h2']:
                if text is None:
                    break
                else:
                    text = text.previous_sibling               

            if text is None:
                pass
            else:
                title_hierarchy.append(text.find(class_='mw-headline').text)
    
    if len(title_hierarchy) == 0:
        return ''
    else:
        tmp = ' -- '.join(title_hierarchy[::-1])
        return normalize(tmp)

def get_section_text(r):
    text = r.previous_sibling
    section_text = ''
    while text is not None:
        if text == '\n':
            text = text.previous_sibling
        elif text.name in ['h1', 'h2', 'h3', 'h4']:
            break
        else:
            tmp = text.text
            if tmp:
                mask = ['note', 'indicate', 'incomplete', 'source', 'reference']
                if  any([_ in tmp.lower() for _ in mask]):
                    tmp = ''
                else:
                    tmp = normalize(tmp)
                    if section_text:
                        section_text = tmp + ' ' + section_text
                    else:
                        section_text = tmp
            text = text.previous_sibling
    return section_text

def normalize(string):
    string = string.strip().replace('\n', ' ')
    return tokenize(string)

def harvest_tables(f_name):
    results = []
    with open(f_name, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
        #rs = soup.find_all(class_='wikitable sortable')
        tmp = soup.find_all(class_='wikitable')
        rs = []
        rest = []
        for _ in tmp:
            if _['class'] == ['wikitable', 'sortable']:
                rs.append(_)
            else:
                rest.append(_)
        rs = rs + rest
        for it, r in enumerate(rs):
            heads = []
            rows = []
            replicate = {}
            for i, t_row in enumerate(r.find_all('tr')):
                if i == 0:
                    for h in t_row.find_all(['th', 'td']):
                        h = remove_ref(h)
                        if len(h.find_all('a')) > 0:
                            heads.append(process_link(h))
                        else:
                            heads.append(([h.get_text(separator=" ").strip()], [None]))
                else:
                    row = []
                    for h in t_row.find_all(['th', 'td']):
                        col_idx = len(row)
                        if col_idx in replicate:
                            row.append(replicate[col_idx][1])
                            replicate[col_idx][0] = replicate[col_idx][0] - 1
                            if replicate[col_idx][0] == 0:
                                del replicate[col_idx]

                        h = remove_ref(h)
                        if len(h.find_all('a')) > 0:
                            tmp = process_link(h)
                        else:
                            tmp = ([h.get_text(separator=" ").strip()], [None])

                        if 'rowspan' in h.attrs:
                            # Identify row span cases
                            try:
                                num = int(h['rowspan'])
                                replicate[len(row)] = [num - 1, tmp]
                            except Exception:
                                pass
                        row.extend(tmp)

                    if all([len(cell[0]) == 0 for cell in row]):
                        continue
                    else:
                        rows.append(row)

            rows = rows[:20]
            if any([len(row) != len(heads) for row in rows]) or len(rows) < default_setting['miniumu_row']:
                continue
            else:
                try:
                    section_title = get_section_title(r)
                except Exception:
                    section_title = ''
                try:
                    section_text = get_section_text(r)
                except Exception:
                    section_text = ''
                title = soup.title.string
                title = re.sub(' - Wikipedia', '', title)
                url = 'https://en.wikipedia.org/wiki/{}'.format('_'.join(title.split(' ')))
                table_name = os.path.split(f_name)[1].replace('.html', '')
                uid = table_name + "_{}".format(it)
                results.append({'url': url, 'title': title, 'header': heads, 'data': rows,
                                'section_title': section_title, 'section_text': section_text,
                                'uid': uid})
    return results

def inplace_postprocessing(tables):
    deletes = []
    for i, table in enumerate(tables):
        # Remove sparse columns
        to_remove = []
        for j, h in enumerate(table['header']):
            if 'Coordinates' in h[0][0] or 'Image' in h[0][0]:
                to_remove.append(j)
                continue
            
            count = 0
            total = len(table['data'])
            for d in table['data']:
                if d[j][0][0] != '':
                    count += 1
            
            if count / total < 0.5:
                to_remove.append(j)
        
        bias = 0
        for r in to_remove:
            del tables[i]['header'][r - bias]
            for _ in range(len(table['data'])):
                del tables[i]['data'][_][r - bias]
            bias += 1
        
        # Remove sparse rows
        to_remove = []
        for k in range(len(table['data'])):
            non_empty = [1 if _[0][0] != '' else 0 for _ in table['data'][k]]
            if sum(non_empty) < 0.5 * len(non_empty):
                to_remove.append(k)
        
        bias = 0
        for r in to_remove:        
            del tables[i]['data'][r - bias]
            bias += 1
        
        if len(table['header']) > default_setting['max_header']:
            deletes.append(i)
        elif len(table['header']) <= default_setting['min_header']:
            deletes.append(i)
        else:
            count = 0
            total = 0
            for row in table['data']:
                for cell in row:
                    if len(cell[0][0]) != '':
                        if cell[1] == [None]:
                            count += 1                    
                        total += 1
            if count / total >= default_setting['ratio']:
                deletes.append(i)

    print('out of {} tables, {} need to be deleted'.format(len(tables), len(deletes)))

    bias = 0
    for i in deletes:
        del tables[i - bias]
        bias += 1

def get_summary(page_title):
    original_title = copy.copy(page_title)
    if page_title.startswith('/wiki/'):
        page_title = page_title[6:]
    page_title = urllib.parse.unquote(page_title)
    if '/wiki/' + page_title in dictionary:
        return dictionary['/wiki/' + page_title]
    elif page_title in cache:
        return cache[page_title]
    else:
        if page_title in redirect['forward']:
            page_title = redirect['forward'][page_title]
            if page_title in cache:
                return cache[page_title]
            else:
                return download_summary(original_title)
        else:
            return download_summary(original_title)

def download_summary(page):
    if page.startswith('https'):
        pass
    elif page.startswith('/wiki'):
        page = 'https://en.wikipedia.org{}'.format(page)
    else:
        page = 'https://en.wikipedia.org/wiki/{}'.format(page)
    
    r = http.request('GET', page)
    if r.status == 200:
        data = r.data.decode('utf-8')
        data = data.replace('</p><p>', ' ') 
        soup = BeautifulSoup(data, 'html.parser')
        div = soup.body.find("div", {"class": "mw-parser-output"})
        if div:
            children = div.findChildren("p" , recursive=False)
            summary = 'N/A'
            for child in children:
                if child.get_text().strip() != "":
                    html = str(child)
                    html = html[html.index('>') + 1:].strip()
                    if not html.startswith('<'):
                        summary = child.get_text(separator=" ").strip()
                        break
                    elif html.startswith('<a>') or html.startswith('<b>') or \
                            html.startswith('<i>') or html.startswith('<a ') or html.startswith('<br>'):
                        summary = child.get_text(separator=" ").strip()
                        break
                    else:
                        continue
            return summary
        else:
            return 'N/A'
    elif r.status == 429:
        time.sleep(1)
        return download_summary(page)
    elif r.status == 404:
        return 'N/A'
    else:
        raise ValueError("return with code {}".format(r.status))

def crawl_hyperlinks(table):
    dictionary = {}
    for cell in table['header']:
        if cell[1]:
            for tmp in cell[1]:
                if tmp and tmp not in dictionary:                
                    summary = get_summary(tmp)
                    dictionary[tmp] = summary
        
    for row in table['data']:
        for cell in row:
            if cell[1]:
                for tmp in cell[1]:
                    if tmp and tmp not in dictionary:
                        summary = get_summary(tmp)
                        dictionary[tmp] = summary
    # Getting page summary
    index = table['url'].index('/wiki/') + 6
    name = table['url'][index:]
    try:
        summary = get_summary(name)
    except Exception:
        summary = 'N/A'
    dictionary[name] = summary
    return dictionary

def clean_cell_text(string):
    string = string.replace('"', '')
    string = string.rstrip('^')
    string = string.replace('–', '-')
    #string = re.sub(r'\b([0-9]{4})-', r'\1 - ', string)    
    #string = re.sub(r'^([0-9]{1,2})-([0-9]{1,2})$', r'\1 - \2', string)
    #string = re.sub(r'^([0-9]{1,2})-([0-9]{1,2})-([0-9]{1,2})$', r'\1 - \2 - \3', string)
    string = string.replace('( ', '(')
    string = string.replace(' )', ')')
    string = string.replace('"', '')
    string = string.replace(u"\u00a0", u' ')
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
    return string

def tokenization_tab(f_n):
    with open(f_n) as f:
        table = json.load(f)
    
    for row_idx, row in enumerate(table['data']):
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[0]):
                if ent:
                    table['data'][row_idx][col_idx][0][i] = tokenize(ent, True)
                if table['data'][row_idx][col_idx][1][i]:
                    table['data'][row_idx][col_idx][1][i] = urllib.parse.unquote(table['data'][row_idx][col_idx][1][i])
    
    for col_idx, header in enumerate(table['header']):
        for i, ent in enumerate(header[0]):
            if ent:
                table['header'][col_idx][0][i] = tokenize(ent, True)
            if table['header'][col_idx][1][i]:
                table['header'][col_idx][1][i] = urllib.parse.unquote(table['header'][col_idx][1][i])

    f_n = f_n.replace('/tables/', '/tables_tok/')
    with open(f_n, 'w') as f:
        json.dump(table, f, indent=2)

def tokenization_req(f_n):
    with open(f_n) as f:
        request_document = json.load(f)

    for k, v in request_document.items():
        sents = tokenize(v)
        request_document[k] = sents

    f_n = f_n.replace('/request/', '/request_tok/')
    with open(f_n, 'w') as f:
        json.dump(request_document, f, indent=2)

def recover(string):
    string = string[6:]
    string = string.replace('_', ' ')
    return string
    
def clean_text(string):
    #if "Initial visibility" in string:
    #    return recover(k)
    position = string.find("mw-parser-output")
    if position != -1:
        left_quote = position - 1
        while left_quote >= 0 and string[left_quote] != '(':
            left_quote -= 1
        right_quote = position + 1
        while right_quote < len(string) and string[right_quote] != ')':
            right_quote += 1
        
        string = string[:left_quote] + " " + string[right_quote + 1:]
        
        position = string.find("mw-parser-output")
        if position != -1:
            right_quote = position + 1
            while right_quote < len(string) and string[right_quote] != '\n':
                right_quote += 1
            string = string[:position] + string[right_quote + 1:]
    
    string = string.replace(u'\xa0', u' ')
    string = string.replace('\ufeff', '')
    string = string.replace(u'\u200e', u' ')
    string = string.replace('–', '-')
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
    
    string = string.replace(u'"', u'')
    string = re.sub(r'[\n]+', '\n', string)
    
    string = re.sub(r'\.+', '.', string)
    string = re.sub(r' +', ' ', string)
    
    #string = re.sub(r"'+", "'", string)
    #string = string.replace(" '", " ")
    #string = string.replace("' ", " ")
    string = filter_firstKsents(string, 12)
    
    return string
    
def tokenize_and_clean_text(kv):
    k, v = kv
    v = clean_text(v)
    v = tokenize(v)
    return k, v

if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    pool = Pool(cores)
    print("Initializing the pool of cores")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists('{}/tables'.format(output_folder)):
        os.mkdir('{}/tables'.format(output_folder))
    if not os.path.exists('{}/request'.format(output_folder)):
        os.mkdir('{}/request'.format(output_folder))
    
    if '1' in steps:
        # Step1: Harvesting the tables
        rs = pool.map(harvest_tables, glob.glob('{}/*.html'.format(input_htmls)))
        tables = []
        for r in rs:
            tables = tables + r
        print("Step1: Finishing harvesting the tables")

    if '2' in steps:
        # Step2: Postprocessing the tables
        inplace_postprocessing(tables)
        with open('{}/processed_new_table_postfiltering.json'.format(output_folder), 'w') as f:
            json.dump(tables, f, indent=2)
        print("Step2: Finsihing postprocessing the tables")

    if '3' in steps:
        # Step3: Getting the hyperlinks
        with open('{}/processed_new_table_postfiltering.json'.format(output_folder), 'r') as f:
            tables = json.load(f)
        print("Total of {} tables".format(len(tables)))
        rs = pool.map(crawl_hyperlinks, tables)
        for r in rs:
            dictionary.update(r)
        for k, v in dictionary.items():
            dictionary[k] = re.sub(r'\[[\d]+\]', '', v).strip()
        merged_unquote = {}
        for k, v in dictionary.items():
            merged_unquote[urllib.parse.unquote(k)] = clean_text(v)
        with open('{}/merged_unquote.json'.format(output_folder), 'w') as f:
            json.dump(merged_unquote, f, indent=2)
        print("Step3: Finishing collecting all the links")

    if '4' in steps:
        # Step 4: distribute the tables into separate files
        with open('{}/processed_new_table_postfiltering.json'.format(output_folder), 'r') as f:
            tables = json.load(f)
        with open('{}/merged_unquote.json'.format(output_folder), 'r') as f:
            merged_unquote = json.load(f)

        for idx, table in enumerate(tables):
            for row_idx, row in enumerate(table['data']):
                for col_idx, cell in enumerate(row):
                    for i, ent in enumerate(cell[0]):
                        if ent:
                            table['data'][row_idx][col_idx][0][i] = clean_cell_text(ent)
            
            for col_idx, header in enumerate(table['header']):
                for i, ent in enumerate(header[0]):
                    if ent:
                        table['header'][col_idx][0][i] = clean_cell_text(ent)

            headers = table['header']
            if headers[0][0] == ['']:
                for i in range(len(table['data'])):
                    del table['data'][i][0]
                del headers[0]
            if any([_[0] == ['Rank'] for _ in headers]):
                if table['data'][0][0][0] == ['']:
                    for i in range(len(table['data'])):
                        if table['data'][i][0][0] == ['']:
                            table['data'][i][0][0] = [str(i + 1)]
            if any([_[0] == ['Place'] for _ in headers]):
                if table['data'][0][0][0] == ['']:
                    for i in range(len(table['data'])):
                        if table['data'][i][0][0] == ['']:
                            table['data'][i][0][0] = [str(i + 1)]

            index = table['url'].index('/wiki/') + 6
            name = table['url'][index:]
            summary = merged_unquote[name]
            table['intro'] = summary
            table['uid'] = urllib.parse.unquote(table['uid'])
            with open('{}/tables/{}.json'.format(output_folder, table['uid']), 'w') as f:
                json.dump(table, f, indent=2)

        print("Step4: Finishing remove unnecessary cells")

    if '5' in steps:
        # Step 5: distribute the request into separate files 
        with open('{}/merged_unquote.json'.format(output_folder), 'r') as f:
            merged_unquote = json.load(f)
        
        def get_request_summary(f_id):
            with open(f_id) as f:
                table = json.load(f)
            local_dict = {}
            table_affected = False
            for i, d in enumerate(table['header']):
                for j, url in enumerate(d[1]):
                    if url:
                        url = urllib.parse.unquote(url)
                        linked_content = merged_unquote[url]
                        if len(linked_content.split(' ')) <= 6:
                            table['header'][i][1][j] = None
                            table_affected = True
                        else:
                            local_dict[url] = linked_content
            for i, row in enumerate(table['data']):
                for j, cell in enumerate(row):
                    for k, url in enumerate(cell[1]):
                        if url:
                            url = urllib.parse.unquote(url)
                            linked_content = merged_unquote[url]
                            if len(linked_content.split(' ')) <= 6:
                                table['data'][i][j][1][k] = None
                                table_affected = True
                            else:
                                local_dict[url] = linked_content
            request_file = f_id.replace('/tables/', '/request/')
            with open(request_file, 'w') as f:
                json.dump(local_dict, f, indent=2)
            if table_affected:
                with open(f_id, 'w') as f:
                    json.dump(table, f, indent=2)
        
        fs = glob.glob('{}/tables/*.json'.format(output_folder))
        for f in fs:
            get_request_summary(f)

        print("Step5: Finishing distributing the requests")

    if '6' in steps:
        # Step 6: tokenize the tables and request
        print("Step6: Starting tokenizing")
        if not os.path.exists('{}/request_tok'.format(output_folder)):
            os.mkdir('{}/request_tok'.format(output_folder))
        if not os.path.exists('{}/tables_tok'.format(output_folder)):
            os.mkdir('{}/tables_tok'.format(output_folder))

        deletes = []
        for f in glob.glob('{}/request/*.json'.format(output_folder)):
            with open(f) as handle:
                request_docs = json.load(handle)

            if len(request_docs) == 0:
                deletes.append(f)
                deletes.append(f.replace('/request/', '/tables/'))
            else:
                if len([v for v in request_docs.values() if len(v) > 5]) > 3:
                    pass
                else:
                    deletes.append(f)
                    deletes.append(f.replace('/request/', '/tables/'))

        print("deleting list has {} items".format(len(deletes)))
        for d in deletes:
            os.remove(d)
        pool.map(tokenization_req, glob.glob('{}/request/*.json'.format(output_folder)))
        pool.map(tokenization_tab, glob.glob('{}/tables/*.json'.format(output_folder)))
        print("Step6: Finishing tokenization")
    
    if '7' in steps:
        # Step7: Generate tables without hyperlinks
        if not os.path.exists('{}/plain_tables_tok'.format(output_folder)):
            os.mkdir('{}/plain_tables_tok'.format(output_folder))
        
        table_set = {}
        for file_name in glob.glob('{}/tables_tok/*.json'.format(output_folder)):
            with open(file_name, 'r') as f:
                table = json.load(f)

            for i, h in enumerate(table['header']):
                full_cell = ' , '.join(table['header'][i][0])
                table['header'][i] = full_cell

            for i, row in enumerate(table['data']):
                for j, cell in enumerate(row):
                    full_cell = ' , '.join(cell[0])
                    table['data'][i][j] = full_cell
            
            file_name = os.path.basename(file_name)
            file_name = os.path.splitext(file_name)[0]
            table_set[file_name] = table
        with open('../data/all_plain_tables.json', 'w') as f:
            json.dump(table_set, f)
        print("Step7: Finished generating plain tables")

    if '8' in steps:
        if os.path.exists('Wikipedia/wiki-intro-with-ents-dict.json'):
            with open('Wikipedia/wiki-intro-with-ents-dict.json', 'r') as f:
                entity_to_intro = json.load(f)

            results = pool.map(tokenize_and_clean_text, entity_to_intro.items())
            results = dict(results)
            
            with open('data/wikipedia_request.json', 'w') as f:
                json.dump(results, f)


    # Wrapping up the results
    pool.close()
    pool.join()

