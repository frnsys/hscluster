from hscluster.graph import build_graph, identify_cliques


def hscluster(sim_mat, per_row_outliers=True, disambiguate=True):
    """
    If `disambiguate=False`, will perform soft clustering
    (i.e. items can belong to multiple clusters), and the predicted labels
    will be lists.
    """
    G = build_graph(sim_mat, per_row_outliers=per_row_outliers)
    cliques = identify_cliques(G, disambiguate=disambiguate)

    # Generate labels for scoring the results
    if disambiguate:
        pred_labels = [-1 for i in range(len(sim_mat))]
        for i, clique in enumerate(cliques):
            for j in clique:
                pred_labels[j] = i
    else:
        pred_labels = [[] for i in range(len(sim_mat))]
        for i, clique in enumerate(cliques):
            for j in clique:
                pred_labels[j].append(i)

    return pred_labels


