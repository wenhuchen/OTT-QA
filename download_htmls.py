import json
import sys
import requests
import urllib.request, urllib.error, urllib.parse
import os

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

from multiprocessing import Pool
with open('processed_table.json') as f:
    data = json.load(f)
pool = Pool(64)
print("Initializing the pool of cores")
pool.map(sub_func, data)
pool.close()
pool.join()
