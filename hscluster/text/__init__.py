from hscluster import hscluster
from hscluster.text.preprocess import preprocess, compute_similarities


def hscluster_docs(docs):
    docs = preprocess(docs)
    sim_mat = compute_similarities(docs)
    return hscluster(sim_mat)
