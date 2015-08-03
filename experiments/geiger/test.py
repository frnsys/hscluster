import os
import json
import shutil
from time import time
from hscluster.text import hscluster_docs

data = json.load(open('/Users/ftseng/Desktop/projects/opennews/projects/geiger/examples/climate_example.json', 'r'))
docs = [d['body'] for d in data if len(d['body']) > 300]
docs = docs[:100]

#data = open('sample.txt', 'r').read()
#docs = []
#for d in data.split('\n'):
    #if d and not d.startswith('ID'):
        #docs.append(d)

print(len(docs))


s = time()
labels = hscluster_docs(docs)
print(labels)
print('Took {0:.2f} seconds'.format(time() - s))

if os.path.exists('output'):
    shutil.rmtree('output')
os.makedirs('output')

for label in range(max(labels) + 1):
    path = os.path.join('output', str(label))
    os.makedirs(path)

for i, label in enumerate(labels):
    doc = docs[i]
    path = os.path.join('output', str(label), '{}.txt'.format(i))
    with open(path, 'w') as f:
        f.write(doc)