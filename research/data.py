import json


def load_truth(datafile):
    """
    Load and prep ground-truth data
    """
    data = json.load(open(datafile, 'r'))

    docs = []
    labels = []
    for i, e in enumerate(data):
        for a in e['articles']:
            docs.append(a['body'])
            labels.append(i)

    return docs, labels, data
