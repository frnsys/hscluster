import json
import random

data = json.load(open('/Users/ftseng/Desktop/projects/opennews/projects/geiger/examples/climate_example.json', 'r'))

docs = [d['body'] for d in data if len(d['body']) > 300]
random.shuffle(docs)
docs = docs[:50]

with open('sample.txt', 'w') as f:
    bodies = ['ID:{}\n{}'.format(i, doc) for i, doc in enumerate(docs)]
    f.write('\n\n\n\n'.join(bodies))