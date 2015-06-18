import sys
import json
from sklearn import metrics
from hscluster import preprocess, build_graph, visualize_graph, hs_cluster


datafile = sys.argv[1]

data = json.load(open(datafile, 'r'))

clusters, articles, true = preprocess(data)
G = build_graph(articles, debug=False)

pred = hs_cluster(data, debug=False)

print('Completeness', metrics.completeness_score(true, pred))
print('Homogeneity', metrics.homogeneity_score(true, pred))
print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))

visualize_graph(G)