from time import time
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict
from sklearn import metrics
from hscluster import hscluster
from hscluster.graph import build_graph
from hscluster.visualize import visualize_graph
from hscluster.text.preprocess import preprocess, compute_similarities

def parse_clusters(fname):
    with open(fname, 'r') as f:
        clusters = ['']
        for line in f.readlines():
            if line.strip() == '---':
                clusters.append('')
                continue
            clusters[-1] += line

        # Do a lot of wrangling (maybe too much)
        ids = set()
        map = defaultdict(lambda: {'labels':[]}) # map comment ids to data
        for i, cluster in enumerate(clusters):
            parts = [p for p in cluster.split('\n') if p]
            # Iterate as triples
            for id, body, keywords in zip(*[iter(parts)]*3):
                id = int(id.replace('ID:', ''))
                ids.add(id)
                map[id]['labels'].append(i)
                map[id]['body'] = body

        ids = list(ids)
        raw_labels = [map[i]['labels'] for i in ids]
        bodies = [map[i]['body'] for i in ids]

        # Some comments may belong to more than one cluster,
        # We create permutations of every possible list of labels
        all_labels = list(product(*raw_labels))

        return bodies, all_labels, raw_labels


def main():
    docs, all_labels, raw_labels = parse_clusters('truth/climate_change_truth.txt')

    print('Number of docs:', len(docs))
    print('Number of different clusterings:', len(all_labels))


    # Just grab first labeling...
    true = all_labels[0]
    true_n_clusters = max(true) + 1


    # Label true clusters on docs for repr
    docs = preprocess(docs)
    for doc, cluster in zip(docs, true):
        doc.cluster = cluster

    sim_mat = compute_similarities(docs)

    s = time()
    pred = hscluster(sim_mat)

    print('Looking for {0} clusters'.format(true_n_clusters))
    print('Found {} clusters'.format(max(pred) + 1))
    print('Took {0:.2f} seconds'.format(time() - s))

    for true in all_labels:
        print('Completeness', metrics.completeness_score(true, pred))
        print('Homogeneity', metrics.homogeneity_score(true, pred))
        print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
        print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))
        print('----')

    return

    print('True clusters')
    true_clusters = [[] for i in range(true_n_clusters)]
    for i, label in enumerate(true):
        true_clusters[label].append(docs[i])
    print(true_clusters)

    compute_similarities(true_clusters[7], debug=True)
    print('---------------------')
    return


    # Compare mean similarity across all documents
    # against mean-mean similarity within known clusters
    tsm = compute_similarities(docs)
    mean_sims = []
    for i, clus in enumerate(true_clusters):
        if len(clus) > 1:
            print('Cluster', i)
            sm = compute_similarities(clus)
            print(sm)
            print('Mean sim:', np.mean(sm))
            mean_sims.append(np.mean(sm))
            print('-----')
    print('Total mean sim', np.mean(tsm))
    print('Co-cluster mean sim', np.mean(mean_sims))

    # Setup documents as the nodes
    print('All cliques (potential clusters)')
    G = build_graph(sim_mat)
    mapping = {i: d for i, d in enumerate(docs)}
    nx.relabel_nodes(G, mapping, copy=False)
    cliques = list(nx.find_cliques(G))
    print(cliques)

    visualize_graph(G)

    return

    s = time()
    pred = hscluster(sim_mat)

    print('Looking for {0} clusters'.format(true_n_clusters))
    print('Found {} clusters'.format(max(pred) + 1))
    print('Took {0:.2f} seconds'.format(time() - s))

    for true in all_labels:
        print('Completeness', metrics.completeness_score(true, pred))
        print('Homogeneity', metrics.homogeneity_score(true, pred))
        print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
        print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))
        print('----')


    clusters = [[] for i in range(max(pred) + 1)]
    for i, label in enumerate(pred):
        clusters[label].append(docs[i])
    print(clusters)



if __name__ == '__main__':
    main()
