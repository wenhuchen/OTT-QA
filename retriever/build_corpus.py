import glob
import json
import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=str, default=None)
args = parser.parse_args()

if args.build == '1,2,3':
    fw = open('data/solr-input.json', 'w')
    posts = []
    for fname in glob.glob('../data/tables_tok/*.json'):
        with open(fname, 'r') as f:
            table = json.load(f)
        title = table['title']
        section_title = table['section_title']
        section_text = table['section_text']
        content = "{} | {} | {}".format(title, section_title, section_text)
        fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    fw.close()
elif args.build == '1,2':
    fw = open('data/solr-input.json', 'w')
    posts = []
    for fname in glob.glob('../data/tables_tok/*.json'):
        with open(fname, 'r') as f:
            table = json.load(f)
        title = table['title']
        section_title = table['section_title']
        content = "{} | {}".format(title, section_title)
        fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    fw.close()
elif args.build == '1,2,4':
    fw = open('data/solr-input.json', 'w')
    posts = []
    for fname in glob.glob('../data/tables_tok/*.json'):
        with open(fname, 'r') as f:
            table = json.load(f)
        title = table['title']
        section_title = table['section_title']
        headers = []
        for h in table['header']:
            headers.append(' '.join(h[0]))
        headers = ' '.join(headers)
        content = "{} | {} | {}".format(title, section_title, headers)
        fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    fw.close()
elif args.build == '1':
    fw = open('data/solr-input.json', 'w')
    posts = []
    for fname in glob.glob('../data/tables_tok/*.json'):
        with open(fname, 'r') as f:
            table = json.load(f)
        title = table['title']
        content = "{}".format(title)
        fw.write(json.dumps({'id': table['uid'], 'text': content}) + '\n')
    fw.close()
elif args.build == 'inverse':
    fw = open('data/solr-input-inv.json', 'w')
    posts = []
    with open('../data/merged_unquote_tok.json', 'r') as f:
        data = json.load(f)

    for k, v in data.items():
        linearized = json.dumps({'id': k, 'text': v})
        fw.write(linearized + '\n')
    fw.close()
