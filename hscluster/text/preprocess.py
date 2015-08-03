import numpy as np
from nytnlp.tokenize.keyword import keyword_tokenizes
from hscluster.models import Document
from hscluster.knowledge import IDF, phrases

idf = IDF('data/nyt/entities_idf.json')
idf_reg = IDF('data/nyt/idf.json')


def preprocess(docs):
    """
    Preprocess a list of raw documents (strings)
    """
    toks = keyword_tokenizes(docs, phrases_model=phrases)

    pdocs = []
    for i, tks in enumerate(toks):
        pdocs.append(Document(i, docs[i], tks))

    return pdocs


def compute_similarities(pdocs, term_sim_ref=None, debug=False):
    """
    Compute the similarity matrix for a list of documents
    """
    sim_mat = np.zeros((len(pdocs), len(pdocs)))

    for i, d in enumerate(pdocs):
        for j, d_ in enumerate(pdocs):
            # Just build the lower triangle
            if i > j:
                sim_mat[i,j] = similarity(d, d_, term_sim_ref=term_sim_ref, debug=debug)

    # Construct the full sim mat from the lower triangle
    return sim_mat + sim_mat.T - np.diag(sim_mat.diagonal())


def similarity(d, d_, term_sim_ref=None, debug=False):
    """
    Compute a similarity score for two documents.

    Optionally pass in a `term_sim_ref` dict-like, which should be able
    to take `term1, term2` as args and return their similarity.
    """
    es = set(d.entities)
    es_ = set(d_.entities)
    e_weight = (len(es) + len(es_) - abs(len(es) - len(es_)))/2
    e_score = sum(idf[t] for t in es & es_)

    toks = set(d.tokens)
    toks_ = set(d_.tokens)
    t_weight = (len(toks) + len(toks_) - abs(len(toks) - len(toks_)))/2

    # If no term similarity reference is passed,
    # look only at surface form overlap (i.e. exact overlap)
    shared_toks = toks & toks_
    overlap = [(t, t, idf_reg[t]) for t in shared_toks]
    t_score = sum(idf_reg[t] for t in shared_toks)
    if term_sim_ref is not None:
        # Double-count exact overlaps b/c we are
        # comparing bidirectional term pairs here
        t_score *= 2
        for toks1, toks2 in [(toks, toks_), (toks_, toks)]:
            for t in toks1 - shared_toks:
                best_match = max(toks2, key=lambda t_: term_sim_ref[t, t_])
                sim = term_sim_ref[t, best_match]
                t_score += sim * ((idf_reg[t] + idf_reg[best_match])/2)
                if sim > 0:
                    overlap.append((t, best_match, sim * ((idf_reg[t] + idf_reg[best_match])/2)))

        # Adjust term weight
        #t_weight /= 2

    t_weight = 1/t_weight if t_weight != 0 else 0
    e_weight = 1/e_weight if e_weight != 0 else 0
    t_score *= t_weight
    e_score *= e_weight

    if debug:
        print('\n-------------------------')
        print((d.id, d_.id))
        print('DOC:', d.id)
        print('DOC:', d_.id)
        print('\tBody 1:', d.body)
        print('\tBody 2:', d_.body)
        print('\tEntities:')
        print('\t\t', es)
        print('\t\t', es_)
        print('\t\tEntity overlap:', es & es_)
        print('\t\tEntity weight:', e_weight)
        print('\t\tEntity score:', e_score)

        print('\tTokens:')
        print('\t\t', toks)
        print('\t\t', toks_)
        print('\t\tToken overlap:', overlap)
        print('\t\tToken weight:', t_weight)
        print('\t\tToken score:', t_score)

        print('\tTotal score:', t_score + e_score)

    return t_score + e_score
