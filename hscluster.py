import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sup.color import cprint
from collections import defaultdict
from nytnlp.tokenize.keyword import keyword_tokenizes
from knowledge import IDF, phrases
from models import Article
from outliers import outliers


idf = IDF('data/nyt_entities_idf.json')
idf_reg = IDF('data/nyt_idf.json')


def preprocess(data):
    """
    Load ground truth clusters
    and pre-process the articles.
    """
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
    toks = keyword_tokenizes(docs, phrases_model=phrases)
    for i, a in enumerate(articles):
        a.tokens = toks[i]

    return clusters, articles, labels


def build_graph(articles, debug=True):
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
                if debug:
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
        overlaps = [TS for l, t, o, a_, s, ts, TS in a.overlaps]
        if debug:
            print('cocluster scores', coclusters)
            print('num cocluster members', len(coclusters))
            print(overlaps)
        for i in outliers(overlaps):
            if a.overlaps[i][0] >= 5: # hard cutoff at at least 5 entity overlaps?
                if debug:
                    cprint('title', a.overlaps[i][3].title)
                    cprint('score', a.overlaps[i][4])
                    cprint('token score', a.overlaps[i][5])
                    cprint('total score', a.overlaps[i][6])
                    cprint('status', a.overlaps[i][1])
                    print([e.name for e in a.overlaps[i][2]])
                graph.add_edge(a, a.overlaps[i][3], weight=a.overlaps[i][6])

    return graph


from collections import Counter
def similarity(a, a_):
    """
    Compute a similarity score for two articles.
    """
    es = set(a.entities)
    es_ = set(a_.entities)
    intersect = es & es_
    ent_freq = Counter(a.entities)
    ent_freq_ = Counter(a_.entities)

    toks = set(a.tokens)
    toks_ = set(a_.tokens)
    token_intersect = toks & toks_
    e_weight = (len(es) + len(es_) - abs(len(es) - len(es_)))/2
    t_weight = (len(toks) + len(toks_) - abs(len(toks) - len(toks_)))/2
    #e_weight = min(len(es), len(es_))
    #t_weight = min(len(toks), len(toks_))
    raw_score = sum(idf[t] for t in intersect)
    #raw_score = sum(idf[t] * (ent_freq[t] + ent_freq_[t]) for t in intersect)
    #raw_score = sum(idf[t] * ((ent_freq[t] + ent_freq_[t])/2) for t in intersect)

    # Ran into an article with no entities
    # (it was improperly extracted)
    try:
        e_weight_ = 1/e_weight
    except ZeroDivisionError:
        e_weight_ = 0
    score = e_weight_ * raw_score

    token_freq = Counter(a.tokens)
    token_freq_ = Counter(a_.tokens)
    raw_token_score = sum(idf_reg[t] for t in token_intersect)
    #raw_token_score = sum(idf_reg[t] * (token_freq[t] + token_freq_[t]) for t in token_intersect)
    #raw_token_score = sum(idf_reg[t] * ((token_freq[t] + token_freq_[t])/2) for t in token_intersect)
    token_score = (1/t_weight) * raw_token_score
    total_score = token_score + score
    return intersect, e_weight_, 1/t_weight, score, token_score, total_score


def visualize_graph(G):
    pos = nx.graphviz_layout(G, prog='fdp')
    edge_labels = dict([((u, v), '{:.2f}'.format(d['weight'])) for u, v, d in G.edges(data=True)])
    edge_colors = [d['weight'] for u, v, d in G.edges(data=True)]
    nx.draw(G, pos, node_color='#A0CBE2', edge_color=edge_colors, width=2, edge_cmap=plt.cm.Blues, with_labels=True, font_size=8, node_size=600)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.show()


def hs_cluster(data, debug=True):
    clusters, articles, true_labels = preprocess(data)
    G = build_graph(articles, debug=debug)
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

    #print(cliques)
    return pred_labels
