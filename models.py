from nytnlp.common import spacy

# For more accurate timing, force spacy to load
spacy('foo bar', entity=True, tag=True, parse=False)


class Entity():
    def __init__(self, name, label):
        self.name = name.strip().lower()
        self.label = label

    def __eq__(self, other):
        return self.name == other.name and self.label == other.label

    def __hash__(self):
        return hash('_'.join([self.name, self.label]))

    def __repr__(self):
        return '{} ({})'.format(self.name, self.label)


class Article():
    def __init__(self, id, body, title, cluster):
        self.id = id
        self.body = body
        self.title = title
        self.cluster = cluster

        res = spacy(body, entity=True, tag=True, parse=False)
        self.entities = [Entity(e.string, e.label_) for e in res.ents]

    def __repr__(self):
        return '{}_{}'.format(self.cluster, self.id)
