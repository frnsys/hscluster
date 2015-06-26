import sys
import networkx as nx
import matplotlib.pyplot as plt
import research
from hscluster.graph import build_graph
from hscluster.preprocess import preprocess, compute_similarities


def visualize_graph(G):
    pos = nx.graphviz_layout(G, prog='fdp')
    edge_labels = dict([((u, v), '{:.2f}'.format(d['weight'])) for u, v, d in G.edges(data=True)])
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=edge_colors, width=2, edge_cmap=plt.cm.Blues, with_labels=True, font_size=8, node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.show()


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
