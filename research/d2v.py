import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.models.word2vec import Vocab
from nltk.tokenize import word_tokenize
from broca.preprocess.clean import clean
from sklearn.cluster import KMeans
from research.kmeans import DetK
from broca.knowledge.phrases import Phrases


print('Loading d2v and phrases models...')
model = Doc2Vec.load('data/nyt/doc2vec/nyt.d2v')
phrases = Phrases.load('data/nyt/bigram_model.phrases')
print('Done Loading')


def d2v_cluster(docs, n_clusters=None):
    """
    Cluster based on doc2vec representations.

    The doc2vec inference parts are adapted from:
    <https://gist.github.com/zseder/4201551d7f8608f0b82b>
    """

    docs = [clean(d) for d in docs]
    docs = [d for d in _doc2vec_doc_stream(docs)]
    n_docs = add_new_labels(docs, model)

    # Add new rows to model.syn0
    n = model.syn0.shape[0]
    model.syn0 = np.vstack((
        model.syn0,
        np.empty((n_docs, model.layer1_size), dtype=np.float32)
    ))

    for i in range(n, n + n_docs):
        np.random.seed(
            np.uint32(model.hashfxn(model.index2word[i] + str(model.seed))))
        a = (np.random.rand(model.layer1_size) - 0.5) / model.layer1_size
        model.syn0[i] = a

    # Set model.train_words to False and model.train_labels to True
    model.train_words = False
    model.train_lbls = True

    # Generate representations for the new documents.
    model.train(docs)
    X = model.syn0[n:]

    # Rule of thumb
    if n_clusters is None:
        dk = DetK(X)
        n_clusters = dk.run(len(docs)//3 + 1) # assume 3 docs to a cluster is most likely

    m = KMeans(n_clusters=n_clusters)
    labels = m.fit_predict(X)

    return labels


def _doc2vec_doc_stream(docs):
    """
    Generator to feed sentences to the dov2vec model.
    """
    for i, doc in enumerate(docs):
        doc = doc.lower()
        tokens = word_tokenize(doc)
        yield LabeledSentence(phrases[tokens], ['SENT_{}'.format(i+1)])


# new labels to self.vocab
def add_new_labels(sentences, model):
    """
    Add new labels (for new docs) to the doc2vec model's `self.vocab`.

    from: <https://gist.github.com/zseder/4201551d7f8608f0b82b>
    """
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
