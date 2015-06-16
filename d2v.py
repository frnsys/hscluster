import json
import math
import numpy as np
from gensim.models import Phrases
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.models.word2vec import Vocab
from nltk.tokenize import word_tokenize, sent_tokenize
from sup.progress import Progress
from nytnlp.clean import clean_doc
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics

def d2v_cluster(emails, n_clusters=None):
    # Vector reps
    print('Loading d2v')
    model = Doc2Vec.load('data/doc2vec/nyt_fulldoc.d2v')
    print('Done Loading')
    docs = [clean_doc(e) for e in emails]

    sents = [s for s in _doc2vec_doc_stream(docs)]
    n_sentences = add_new_labels(sents, model)

    # add new rows to model.syn0
    n = model.syn0.shape[0]
    model.syn0 = np.vstack((
        model.syn0,
        np.empty((n_sentences, model.layer1_size), dtype=np.float32)
    ))

    for i in range(n, n + n_sentences):
        np.random.seed(
            np.uint32(model.hashfxn(model.index2word[i] + str(model.seed))))
        a = (np.random.rand(model.layer1_size) - 0.5) / model.layer1_size
        model.syn0[i] = a

    # Set model.train_words to False and model.train_labels to True
    model.train_words = False
    model.train_lbls = True

    # train
    model.train(sents)
    X = model.syn0[n:]

    if n_clusters is None:
        n_clusters = int(math.sqrt(len(emails)/2))
    print('Looking for {0} clusters'.format(n_clusters))

    s = time()
    m = KMeans(n_clusters=n_clusters)
    labels = m.fit_predict(X)
    print('Took {0:.2f} seconds'.format(time() - s))

    return labels


def _doc2vec_doc_stream(docs):
    """
    Generator to feed sentences to the dov2vec model.
    """
    phrases = Phrases.load('data/bigram_model.phrases')

    n = len(docs)
    i = 0
    p = Progress()
    for doc in docs:
        i += 1
        p.print_progress(i/n)

        # We do minimal pre-processing here so the model can learn
        # punctuation
        doc = doc.lower()

        #for sent in sent_tokenize(doc):
            #tokens = word_tokenize(sent)
            #yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i)])
        tokens = word_tokenize(doc)
        yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i)])


# new labels to self.vocab
def add_new_labels(sentences, model):
    sentence_no = -1
    total_words = 0
    vocab = model.vocab
    #model_sentence_n = len([l for l in vocab if l.startswith("SENT")])
    model_sentence_n = max(int(l.split('_')[-1]) for l in vocab if l.startswith("SENT"))
    n_sentences = 0
    for sentence_no, sentence in enumerate(sentences):
        sentence_length = len(sentence.words)
        for label in sentence.labels:
            label_e = label.split("_")
            label_n = int(label_e[1]) + model_sentence_n
            label = "{0}_{1}".format(label_e[0], label_n)
            total_words += 1
            if label in vocab:
                vocab[label].count += sentence_length
            else:
                vocab[label] = Vocab(count=sentence_length)
                vocab[label].index = len(model.vocab) - 1
                vocab[label].code = [0]
                vocab[label].sample_probability = 1.
                model.index2word.append(label)
                n_sentences += 1
    return n_sentences


def load_truth():
    data = json.load(open('out.json', 'r'))

    articles = []
    labels = []
    for i, e in enumerate(data):
        for a in e['articles']:
            articles.append(a['body'])
            labels.append(i)

    return articles, labels


if __name__ == '__main__':
    articles, true = load_truth()

    pred = d2v_cluster(articles, n_clusters=10)

    print('Completeness', metrics.completeness_score(true, pred))
    print('Homogeneity', metrics.homogeneity_score(true, pred))
    print('Adjusted Mutual Info', metrics.adjusted_mutual_info_score(true, pred))
    print('Adjusted Rand', metrics.adjusted_rand_score(true, pred))
