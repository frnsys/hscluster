from hscluster.preprocess import preprocess, compute_similarities
from hscluster.graph import build_graph, identify_cliques


def hscluster(docs):
    docs = preprocess(docs)
    sim_mat = compute_similarities(docs)
    G = build_graph(sim_mat)
    cliques = identify_cliques(G)

    # Generate labels for scoring the results
    pred_labels = [-1 for i in range(len(sim_mat))]
    for i, clique in enumerate(cliques):
        for j in clique:
            pred_labels[j] = i

    return pred_labels
