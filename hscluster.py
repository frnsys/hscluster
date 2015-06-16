import json
from time import time
import networkx as nx
import numpy as np
from sklearn import metrics

from knowledge import IDF
from models import Article
from nytnlp.tokenize.keyword import keyword_tokenizes
from gensim.models import Phrases
from sup.color import cprint
from collections import defaultdict
import matplotlib.pyplot as plt


idf = IDF('data/nyt_entities_idf.json')
idf_reg = IDF('data/nyt_idf.json')
bigram = Phrases.load('data/bigram_model.phrases')


def load_truth(datafile):
    data = json.load(open(datafile, 'r'))

    articles = []
    clusters = []
    labels = []

    id = 0
    for i, e in enumerate(data):
        arts = []
        for a in e['articles']:
            art = Article(id, a['body'], a['title'], i)
            arts.append(art)
            labels.append(i)
            id += 1
        for a in arts:
            a.cocluster = arts

        clusters.append(arts)
        articles += arts

    docs = [a.body for a in articles]
    toks = keyword_tokenizes(docs, phrases_model=bigram)
    for i, a in enumerate(articles):
        a.tokens = set(toks[i])

    return clusters, articles, labels


def build_graph(articles):
    """
    Build the graph out of the articles and their overlaps.
    """
    graph = nx.Graph()
    graph.add_nodes_from(articles)

    for a in articles:
        a.overlaps = []
        coclusters = []
        for a_ in articles:
            if a != a_:
                intersect, e_weight, t_weight, score, token_score, total_score = similarity(a, a_)
                print('ARTICLE', a_.title)
                print('\tentity weight', e_weight)
                print('\ttoken weight', t_weight)
                print('\tfinal score', score)
                print('\tfinal token score', token_score)
                cprint('\ttotal score', total_score)
                cprint('\tstatus', a_ in a.cocluster)
                a.overlaps.append((len(intersect), a_ in a.cocluster, intersect, a_, score, token_score, total_score))
                if a_ in a.cocluster:
                    coclusters.append(score)
        print('cocluster scores', coclusters)
        print('num cocluster members', len(coclusters))
        overlaps = [TS for l, t, o, a_, s, ts, TS in a.overlaps]
        print(overlaps)
        oi = np.where(mad_based_outlier(np.array(overlaps)))
        print(oi)
        for i in outliers(overlaps):
            if a.overlaps[i][0] >= 5: # hard cutoff at at least 5 entity overlaps?
                cprint('title', a.overlaps[i][3].title)
                cprint('score', a.overlaps[i][4])
                cprint('token score', a.overlaps[i][5])
                cprint('total score', a.overlaps[i][6])
                cprint('status', a.overlaps[i][1])
                print([e.name for e in a.overlaps[i][2]])
                graph.add_edge(a, a.overlaps[i][3], weight=a.overlaps[i][4])

    return graph


def similarity(a, a_):
    """
    Compute a similarity score for two articles.
    """
    es = set(a.entities)
    es_ = set(a_.entities)
    intersect = es & es_
    token_intersect = a.tokens & a_.tokens
    e_weight = (len(es) + len(es_) - abs(len(es) - len(es_)))/2
    t_weight = (len(a.tokens) + len(a_.tokens) - abs(len(a.tokens) - len(a_.tokens)))/2
    raw_score = sum(idf[t] for t in intersect)

    try:
        e_weight_ = 1/e_weight
    except ZeroDivisionError:
        e_weight_ = 0
    score = e_weight_ * raw_score

    raw_token_score = sum(idf_reg[t] for t in token_intersect)
    token_score = (1/t_weight) * raw_token_score
    total_score = token_score + score
    return intersect, e_weight_, 1/t_weight, score, token_score, total_score


def jump_outliers(values):
    values = sorted(values)
    diffs = [y-x for x,y in zip(values, values[1:])]

    avg_diffs = []
    for i in range(len(diffs)):
        avg = sum(diffs[:i])/(i+1)
        avg_diffs.append(diffs[i]/(avg+1))

    return list(range(np.argmax(avg_diffs), len(values)))


def outliers(values, thresh=2.):
    # Quartiles
    q = np.percentile(values, np.arange(0, 100, 25))
    q1 = q[0]
    q3 = q[2]
    interquartile_range = q3 - q1
    outlier_thresh = q3 + thresh * interquartile_range
    print('\toutlier thresh', outlier_thresh)

    return [i for i, v in enumerate(values) if v > outlier_thresh]


def mad_based_outlier(points, thresh=2.5):
    """
    Source: <http://stackoverflow.com/a/22357811>
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def visualize_graph(G):
    pos = nx.graphviz_layout(G, prog='fdp')
    edge_labels = dict([((u, v), '{:.2f}'.format(d['weight'])) for u, v, d in G.edges(data=True)])
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=edge_colors, width=2, edge_cmap=plt.cm.Blues, with_labels=True, font_size=8, node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.show()


def cluster(articles, graph):
    cliques = list(nx.find_cliques(G))

    # Count how many cliques an article belongs to
    article_membership = defaultdict(int)
    for c in cliques:
        for a in c:
            article_membership[a] += 1

    # "Disputed" articles are those that belong to multiple cliques
    disputed = []
    for k, v in article_membership.items():
        if v > 1:
            disputed.append(k)

    # We keep disputed articles in the clique which they have the
    # strongest connection to, and remove them from their other cliques
    for a in disputed:
        cls = [c for c in cliques if a in c]
        scores = []
        for c in cls:
            nodes = [n for n in c if n not in disputed]
            if not nodes:
                scores.append(0)
            else:
                sumscore = sum(G.edge[a][n]['weight'] for n in nodes)
                scores.append(sumscore)
        idx = np.argmax(scores)
        for c in cliques:
            if a in c and c != cls[idx]:
                c.remove(a)

    cliques = [c for c in cliques if c]

    # Generate labels for scoring the results
    pred_labels = []
    for a in articles:
        for i, c in enumerate(cliques):
            if a in c:
                pred_labels.append(i)
                break

    return pred_labels


if __name__ == '__main__':
    s = time()
    clusters, articles, true_labels = load_truth('data/truth/out.json')
    print('Took {:.2f}s'.format(time() - s))

    #s = time()
    G = build_graph(articles)
    pred_labels = cluster(articles, G)
    print('Took {:.2f}s'.format(time() - s))

    print('Completeness', metrics.completeness_score(true_labels, pred_labels))
    print('Homogeneity', metrics.homogeneity_score(true_labels, pred_labels))
    print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true_labels, pred_labels))
    print('Adjusted Rand', metrics.adjusted_rand_score(true_labels, pred_labels))
