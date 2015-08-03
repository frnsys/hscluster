import research
from research.text import Vectorizer
from nytnlp.clean import clean_doc
from time import time
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from hscluster import hscluster
from hscluster.text import hscluster_docs
from hscluster.preprocess import preprocess


if __name__ == '__main__':
    datafiles = [
        'research/data/10E.json',
        'research/data/20E.json',
        'research/data/30E.json'
    ]

    for datafile in datafiles:
        print('\n{}'.format('-'*20))
        print(datafile)
        print('-'*20)

        docs, true, data = research.load_truth(datafile)
        true_n_clusters = max(true) + 1

        methods = [
            (research.dbscan_cluster, {'eps': None})
        ]

        # For larger number of clusters,
        # the KMeans parameter estimation runs indefinitely
        if true_n_clusters <= 20:
            methods += [
                (research.d2v_cluster, {'n_clusters': None}),
                (research.kmeans_cluster, {'n_clusters': None}),
            ]

        for method, kwargs in methods:
            print('\n--------', method.__name__)
            s = time()
            pred = method(docs, **kwargs)
            print('Looking for {0} clusters'.format(true_n_clusters))
            print('Found {} clusters'.format(max(pred) + 1))
            print('Took {0:.2f} seconds'.format(time() - s))

            print('Completeness', metrics.completeness_score(true, pred))
            print('Homogeneity', metrics.homogeneity_score(true, pred))
            print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
            print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))


        print('\n--------', hscluster.__name__)
        s = time()
        vectr = Vectorizer()
        cdocs = [clean_doc(d) for d in docs]
        vecs = vectr.vectorize(cdocs, train=True)
        sim_mat = pdist(vecs.todense())
        sim_mat = squareform(sim_mat)
        pred = hscluster(sim_mat)
        print('Looking for {0} clusters'.format(true_n_clusters))
        print('Found {} clusters'.format(max(pred) + 1))
        print('Took {0:.2f} seconds'.format(time() - s))

        print('Completeness', metrics.completeness_score(true, pred))
        print('Homogeneity', metrics.homogeneity_score(true, pred))
        print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
        print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))


        print('\n--------', hscluster_docs.__name__)
        s = time()
        pred = hscluster_docs(docs)
        print('Looking for {0} clusters'.format(true_n_clusters))
        print('Found {} clusters'.format(max(pred) + 1))
        print('Took {0:.2f} seconds'.format(time() - s))

        print('Completeness', metrics.completeness_score(true, pred))
        print('Homogeneity', metrics.homogeneity_score(true, pred))
        print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
        print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))

        # Print out hscluster cluster assignments
        docs = preprocess(docs)
        clusters = [[] for i in range(max(pred) + 1)]
        for doc, cluster in zip(docs, true):
            doc.cluster = cluster
        for i, label in enumerate(pred):
            clusters[label].append(docs[i])
        print(clusters)
