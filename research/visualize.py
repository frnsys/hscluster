import sys
import research
import networkx as nx
from hscluster.graph import build_graph
from hscluster.visualize import visualize_graph
from hscluster.text.preprocess import preprocess, compute_similarities


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
    else:
        datafile = 'research/data/10E.json'

    docs, true, data = research.load_truth(datafile)

    docs = preprocess(docs)

    for doc, cluster in zip(docs, true):
        doc.cluster = cluster

    sim_mat = compute_similarities(docs)
    G = build_graph(sim_mat)

    # Setup documents as the nodes
    mapping = {i: d for i, d in enumerate(docs)}
    nx.relabel_nodes(G, mapping, copy=False)

    visualize_graph(G)
