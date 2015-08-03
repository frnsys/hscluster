import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from hscluster.outliers import outlier_indices, outlier_threshold


def filter_outliers(outlier_thresh=None):
    """
    Zero out indices of non-outliers in a row
    """
    def per_row(row):
        outlier_idx = outlier_indices(row, outlier_thresh=outlier_thresh)

        # Get indices of non-outliers and zero them out
        non_outlier_mask = np.ones(len(row), np.bool)
        non_outlier_mask[outlier_idx] = False
        row[non_outlier_mask] = 0
        return row
    return per_row


def disambiguate_cliques(G, cliques):
    """
    Disambiguate cliques so that nodes only belong to one clique
    """
    memberships = defaultdict(list)

    for i, cliq in enumerate(cliques):
        for mem in cliq:
            memberships[mem].append(i)

    # Only look at members that belong to multiple cliques
    memberships = {mem: cliqs for mem, cliqs in memberships.items() if len(cliqs) > 1}

    print('Need to disambiguate members:', len(memberships))

    # Identify the strongest clique for each ambiguous member,
    # remove it from their clique list
    print('Computing strongest cliques...')
    for mem, cliqs in memberships.items():
        #i_best = max(cliqs, key=lambda i: sum(G.edge[mem][n]['weight'] for n in cliques[i] if n != mem))
        i_best = max(cliqs, key=lambda i: np.mean([G.edge[mem][n]['weight'] for n in cliques[i] if n != mem]))
        cliqs.remove(i_best)

    # Remove ambiguous members from their non-best cliques
    print('Pruning memberships...')
    for mem, cliqs in memberships.items():
        for i in cliqs:
            cliques[i].remove(mem)

    # Keep non-empty cliques
    cliques = [c for c in cliques if c]

    return cliques


def prune_cliques(G, cliques):
    strengths = []
    for cliq in cliques:
        weights = []
        for m1, m2 in combinations(cliq, 2):
            weights.append(G.edge[m1][m2]['weight'])
        strengths.append(np.mean(weights))
    threshold = np.mean(strengths)

    survivors = []
    for i, cliq in enumerate(cliques):
        if strengths[i] >= threshold:
            survivors.append(cliq)
    return survivors


def build_graph(sim_mat, per_row_outliers=True):
    if not per_row_outliers:
        # We only need the lower triangle, not including the diagonal
        sims = sim_mat[np.tril_indices(sim_mat.shape[0], k=-1)]
        outlier_thresh = outlier_threshold(sims)
        print('~~~~~~~~~')
        print('~~~~~~~~~')
        print(outlier_thresh)
        print('~~~~~~~~~')
        print('~~~~~~~~~')
    else:
        outlier_thresh = None

    # Zero-out non-outliers in the sim mat
    # This gives us a sim mat which works as an adj mat
    adj_mat = np.apply_along_axis(filter_outliers(outlier_thresh=outlier_thresh), 1, sim_mat)

    # Build graph, w/ similarities as edge weights
    G = nx.from_numpy_matrix(adj_mat)
    return G


def identify_cliques(G, disambiguate=True):
    """
    Identify cliques, and optionally disambiguate them.
    Set `disambiguate=False` if you want soft clustering (multi-membership).
    """
    print('Identifying cliques...')
    cliques = list(nx.find_cliques(G))

    if disambiguate:
        print('Disambiguating cliques...')
        cliques = disambiguate_cliques(G, cliques)

    print('Done with cliques')
    return cliques
