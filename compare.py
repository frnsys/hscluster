import sys
import json
from time import time
from sklearn import metrics

from d2v import d2v_cluster
from kmeans import kmeans_cluster
from dbscan import dbscan_cluster
from hscluster import hs_cluster


def load_truth(datafile):
    data = json.load(open(datafile, 'r'))

    articles = []
    labels = []
    for i, e in enumerate(data):
        for a in e['articles']:
            articles.append(a['body'])
            labels.append(i)

    return articles, labels, data


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
    else:
        datafile = 'data/truth/out.json'

    articles, true, data = load_truth(datafile)
    true_n_clusters = max(true) + 1

    methods = [
        #(d2v_cluster, {'n_clusters': None}),
        #(kmeans_cluster, {'n_clusters': None}),
        (dbscan_cluster, {'eps': None})
    ]

    for method, kwargs in methods:
        print('\n--------', method.__name__)
        s = time()
        pred = method(articles, **kwargs)
        print('Took {0:.2f} seconds'.format(time() - s))

        print('Completeness', metrics.completeness_score(true, pred))
        print('Homogeneity', metrics.homogeneity_score(true, pred))
        print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
        print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))


    print('\n--------', hs_cluster.__name__)
    s = time()
    pred = hs_cluster(data, debug=False)
    print('Found {} clusters'.format(max(pred) + 1))
    print('Took {0:.2f} seconds'.format(time() - s))

    print('Completeness', metrics.completeness_score(true, pred))
    print('Homogeneity', metrics.homogeneity_score(true, pred))
    print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
    print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))
