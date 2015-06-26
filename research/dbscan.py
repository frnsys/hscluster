import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from nytnlp.clean import clean_doc
from research.text import Vectorizer


def dbscan_cluster(docs, eps=None):
    vectr = Vectorizer()
    docs = [clean_doc(d) for d in docs]
    vecs = vectr.vectorize(docs, train=True)

    if eps is None:
        dist_mat = build_dist_mat(vecs)
        eps = estimate_eps(dist_mat)[0]

    m = DBSCAN(min_samples=3, metric='euclidean', eps=eps)
    labels = m.fit_predict(vecs)
    return labels


def build_dist_mat(vecs):
    dm_ = pdist(vecs.todense(), metric='euclidean')
    dm_ = squareform(dm_)
    return dm_


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
        difs = [(x, y,
                (y - x)) for x, y in zip(dists, dists[1:])]
        eps_candidate, _, jump = max(difs, key=lambda x: x[2])

        estimates.append(eps_candidate)
        return sorted(estimates)
