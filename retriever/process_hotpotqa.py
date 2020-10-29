import os
import glob
import bz2
import json

all_files = []
root_dir = 'Wikipedia/enwiki-20171001-pages-meta-current-withlinks-abstracts/'
for folder in os.listdir(root_dir):
	if len(folder) == 2:
		for file_name in glob.glob(os.path.join('{}/{}/*.bz2'.format(root_dir, folder))):
			all_files.append(file_name)
print("Done gathering all file names = {}".format(len(all_files)))


def gather_info(file_name):
    pair = []
    with bz2.open(file_name) as f:
        for line in f:
            line = json.loads(line.decode('utf-8'))
            pair.append(('/wiki/' + line['title'].replace(' ', '_'), " ".join(line['text'])))
    return pair

from multiprocessing import Pool
pool = Pool(64)

outputs = pool.map(gather_info, all_files)
dictionary = {}
for output in outputs:
    dictionary.update(dict(output))
print("Done mapping all the functions")

with open('Wikipedia/wiki-intro-with-ents-dict-hotpot.json', 'w') as f:
    json.dump(dictionary, f)
