import json
import random
from datetime import datetime

data = json.load(open('/Users/ftseng/Desktop/projects/opennews/projects/supplychain/data/source/argos/articles.json', 'r'))

start = datetime(year=2014, month=12, day=4)
end   = datetime(year=2014, month=12, day=5)

docs = []
for a in data:
    date = datetime.fromtimestamp(a['created_at'])

    if start < date < end:
        docs.append(a['text'])

random.shuffle(docs)
docs = docs[:50]

with open('sample.txt', 'w') as f:
    f.write('\n\n\n\n'.join(docs))