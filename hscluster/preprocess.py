import numpy as np
from nytnlp.tokenize.keyword import keyword_tokenizes
from hscluster.models import Document
from hscluster.knowledge import IDF, phrases


idf = IDF('data/nyt_entities_idf.json')
idf_reg = IDF('data/nyt_idf.json')


def preprocess(docs):
    """
    Preprocess a list of raw documents (strings)
    """
    toks = keyword_tokenizes(docs, phrases_model=phrases)

    pdocs = []
    for i, tks in enumerate(toks):
        pdocs.append(Document(i, docs[i], tks))

    return pdocs


def compute_similarities(pdocs):
    """
    Compute the similarity matrix for a list of documents
    """
    sim_mat = np.zeros((len(pdocs), len(pdocs)))

    for i, d in enumerate(pdocs):
        for j, d_ in enumerate(pdocs):
            # Just build the lower triangle
            if i > j:
                sim_mat[i,j] = similarity(d, d_)

    # Construct the full sim mat from the lower triangle
    return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())


def similarity(d, d_):
    """
    Compute a similarity score for two documents.
    """
    es = set(d.entities)
    es_ = set(d_.entities)

    # Require at least 5 entity overlap (y/n?)
    if len(es & es_) < 5:
        return 0

    e_weight = (len(es) + len(es_) - abs(len(es) - len(es_)))/2
    e_score = sum(idf[t] for t in es & es_)

    toks = set(d.tokens)
    toks_ = set(d_.tokens)
    t_weight = (len(toks) + len(toks_) - abs(len(toks) - len(toks_)))/2
    t_score = sum(idf_reg[t] for t in toks & toks_)
    t_score *= (1/t_weight)

    # Ran into an article with no entities
    # (it was improperly extracted)
    try:
        e_weight_ = 1/e_weight
    except ZeroDivisionError:
        e_weight_ = 0
    e_score *= e_weight_

    return t_score + e_score
