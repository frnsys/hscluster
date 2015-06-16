import json
import numpy as np
from sklearn import metrics

import math
from sklearn.cluster import KMeans, DBSCAN
from nytnlp.clean import clean_doc
from text import Vectorizer
from time import time
from scipy.spatial.distance import pdist, squareform

# adapted from
# <https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/>
class DetK():
    def __init__(self, X):
        self.X = X

    def fK(self, thisk, Skm1=0):
        X = self.X
        Nd = X.shape[1]
        a = lambda k, Nd: 1 - 3./(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6.

        m = KMeans(n_clusters=thisk)
        labels = m.fit_predict(X)
        mu = m.cluster_centers_
        clusters = [[] for _ in range(max(labels) + 1)]
        for i, l in enumerate(labels):
            clusters[l].append(X[i])

        Sk = sum([np.linalg.norm(mu[i]-c)**2 \
                 for i in range(thisk) for c in clusters[i]])
        if thisk == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk/(a(thisk,Nd)*Skm1)
        return fs, Sk

    def run(self, maxk):
        ks = range(1,maxk)
        fs = np.zeros(len(ks))
        # Special case K=1
        fs[0], Sk = self.fK(1)
        # Rest of Ks
        for k in ks[1:]:
            fs[k-1], Sk = self.fK(k, Skm1=Sk)
        print(fs)
        return np.argmin(fs) + 1


def cluster(emails, approach, n_clusters=None, eps=0.1):
    # Vector reps
    s = time()
    vectr = Vectorizer()
    docs = [clean_doc(e) for e in emails]
    vecs = vectr.vectorize(docs, train=True)

    dk = DetK(vecs)
    est_k = dk.run(10)
    print(est_k)

    # Default to rule-of-thumb
    if n_clusters is None:
        n_clusters = int(math.sqrt(len(emails)/2))
    print('Looking for {0} clusters'.format(n_clusters))

    if approach == 'kmeans':
        m = KMeans(n_clusters=n_clusters)
    elif approach == 'dbscan':
        m = DBSCAN(min_samples=3, metric='euclidean', eps=eps)
    labels = m.fit_predict(vecs)
    print('Took {0:.2f} seconds'.format(time() - s))

    return labels


def build_dist_mat(docs):
    vectr = Vectorizer()
    docs = [clean_doc(e) for e in docs]
    vecs = vectr.vectorize(docs, train=True)
    dm_ = pdist(vecs.todense(), metric='euclidean')
    dm_ = squareform(dm_)
    return dm_


def load_truth():
    data = json.load(open('data/truth/out.json', 'r'))

    articles = []
    labels = []
    for i, e in enumerate(data):
        for a in e['articles']:
            articles.append(a['body'])
            labels.append(i)

    return articles, labels


def estimate_eps(dist_mat, n_closest=5):
    """
    Estimates possible eps values (to be used with DBSCAN)
    for a given distance matrix by looking at the largest distance "jumps"
    amongst the `n_closest` distances for each item.

    Tip: the value for `n_closest` is important - set it too large and you may only get
    really large distances which are uninformative. Set it too small and you may get
    premature cutoffs (i.e. select jumps which are really not that big).

    TO DO this could be fancier by calculating support for particular eps values,
    e.g. 80% are around 4.2 or w/e
    """
    dist_mat = dist_mat.copy()

    # To ignore i == j distances
    dist_mat[np.where(dist_mat == 0)] = np.inf
    estimates = []
    for i in range(dist_mat.shape[0]):
        # Indices of the n closest distances
        row = dist_mat[i]
        dists = sorted(np.partition(row, n_closest)[:n_closest])
        difs = [(x,
                 y,
                 (y - x)) for x, y in zip(dists, dists[1:])]
        eps_candidate, _, jump = max(difs, key=lambda x: x[2])

        estimates.append(eps_candidate)
        return sorted(estimates)



if __name__ == '__main__':
    articles, true = load_truth()

    #dm = build_dist_mat(articles)
    #estimates = estimate_eps(dm, n_closest=3)
    #print(estimates)

    #print('True n clusters', max(true) + 1)

    #for e in np.arange(estimates[0] - 0.2, estimates[0] + 0.2, 0.025):
        #print('---------', e)
        #pred = cluster(articles, 'dbscan', n_clusters=10, eps=e)

        #print('Completeness', metrics.completeness_score(true, pred))
        #print('Homogeneity', metrics.homogeneity_score(true, pred))
        #print('Adjusted Mutual Info Score', metrics.adjusted_mutual_info_score(true, pred))
        #print('Adjusted Rand Score', metrics.adjusted_rand_score(true, pred))

    pred = cluster(articles, 'kmeans', n_clusters=10)

    print('Completeness', metrics.completeness_score(true, pred))
    print('Homogeneity', metrics.homogeneity_score(true, pred))
    print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
    print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))