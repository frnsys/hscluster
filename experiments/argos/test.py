import json
from time import time
from datetime import datetime
from hscluster.text import hscluster_docs

data = json.load(open('/Users/ftseng/Desktop/projects/opennews/projects/supplychain/data/source/argos/articles.json', 'r'))

start = datetime(year=2014, month=12, day=4)
end   = datetime(year=2014, month=12, day=5)

articles = []
for a in data:
    date = datetime.fromtimestamp(a['created_at'])

    if start < date < end:
        articles.append({
            'title': a['title'],
            'date': date,
            'body': a['text']
        })


print(len(articles))


s = time()
labels = hscluster_docs([a['body'] for a in articles])
print('Took {0:.2f} seconds'.format(time() - s))