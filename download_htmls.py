import json
import sys
import requests
import urllib.request, urllib.error, urllib.parse
import os
import html
from urllib.parse import quote
import time

from_where = sys.argv[1]

def sub_func(d):
    title = d['title']
    title = '_'.join(title.split(' '))    
    page = 'https://en.wikipedia.org/wiki/{}'.format(title)

    if len(d['data']) > 2 and len(d['data']) < 60 and len(d['data'][0]) >= 3:
        headers = set(d['header'])
        if len(headers) == len(d['header']):
            cols = len(d['header'])
            count = 0
            for g in d['data'][0]:
                if g[1] is not None:
                    count += 1
            if count < 0.1 * cols:
                return
            
            # process if there are enough hyperlinks
            title = d['title']
            title = '_'.join(title.split(' '))

        if not os.path.exists('htmls/{}.html'.format(title)):
            try:
                response = urllib.request.urlopen(page)
                webContent = response.read()
                f = open('htmls/{}.html'.format(title), 'wb')
                f.write(webContent)
                f.close()
            except Exception:
                return

def download_page(title):
    title = quote(title) 
    page = 'https://en.wikipedia.org/wiki/{}'.format(title)
    if not os.path.exists('htmls/{}.html'.format(title)):
        response = urllib.request.urlopen(page)
        webContent = response.read()
        f = open('htmls/{}.html'.format(title), 'wb')
        f.write(webContent)

def direct_download(title):
    try:
        download_page(title)
    except Exception as e:
        if '429' in str(e):
            print("Sleep and retry")
            time.sleep(4)
            try:
                download_page(title)
            except Exception:
                return
        else:
            print(e, title)
            return

if __name__ == '__main__':
    from multiprocessing import Pool
    pool = Pool(64)
    if from_where == 'northwestern': 
        with open('processed_table.json') as f:
            data = json.load(f)
        print("Initializing the pool of cores")
        pool.map(sub_func, data)
    elif from_where == 'tapas':
        with open('tapas_htmls.json', 'r') as f:
            data = json.load(f)
        #for d in data:
        #    direct_download(d)
        pool.map(direct_download, data)
    else:
        raise NotImplementedError
    pool.close()
    pool.join()
