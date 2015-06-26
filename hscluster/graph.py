import numpy as np
import networkx as nx
from hscluster.outliers import outlier_indices
from collections import defaultdict


def filter_outliers(row):
    """
    Zero out indices of non-outliers in a row
    """
    outlier_idx = outlier_indices(row)

    # Get indices of non-outliers and zero them out
    non_outlier_mask = np.ones(len(row), np.bool)
    non_outlier_mask[outlier_idx] = False
    row[non_outlier_mask] = 0
    return row


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

    # Identify the strongest clique for each ambiguous member,
    # remove it from their clique list
    for mem, cliqs in memberships.items():
        i_best = max(cliqs, key=lambda i: sum(G.edge[mem][n]['weight'] for n in cliques[i] if n != mem))
        cliqs.remove(i_best)

    # Remove ambiguous members from their non-best cliques
    for mem, cliqs in memberships.items():
        for i in cliqs:
            cliques[i].remove(mem)

    # Keep non-empty cliques
    cliques = [c for c in cliques if c]

    return cliques


def build_graph(sim_mat):
    # Zero-out non-outliers in the sim mat
    # This gives us a sim mat which works as an adj mat
    adj_mat = np.apply_along_axis(filter_outliers, 1, sim_mat)

    # Build graph, w/ similarities as edge weights
    G = nx.from_numpy_matrix(adj_mat)
    return G


def identify_cliques(G):
    cliques = list(nx.find_cliques(G))
    cliques = disambiguate_cliques(G, cliques)
    return cliques
