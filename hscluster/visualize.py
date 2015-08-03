import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(G):
    pos = nx.graphviz_layout(G, prog='fdp')
    edge_labels = dict([((u, v), '{:.2f}'.format(d['weight'])) for u, v, d in G.edges(data=True)])
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=edge_colors, width=2, edge_cmap=plt.cm.Blues, with_labels=True, font_size=8, node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.show()
