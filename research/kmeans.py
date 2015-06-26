import numpy as np
from sklearn.cluster import KMeans
from nytnlp.clean import clean_doc
from research.text import Vectorizer


class DetK():
    """
    Adapted from
    <https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/>
    """
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
        return np.argmin(fs) + 1


def kmeans_cluster(docs, n_clusters=None):
    vectr = Vectorizer()
    docs = [clean_doc(d) for d in docs]
    vecs = vectr.vectorize(docs, train=True)

    if n_clusters is None:
        dk = DetK(vecs)
        n_clusters = dk.run(len(docs)//3 + 1) # assume 3 docs to a cluster is most likely

    m = KMeans(n_clusters=n_clusters)
    labels = m.fit_predict(vecs)

    return labels