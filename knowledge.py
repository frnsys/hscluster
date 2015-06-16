import json
import math


class IDF():
    def __init__(self, path):
        self._idf = json.load(open(path, 'r'))

    def __getitem__(self, term):
        N = self._idf['_n_docs'] + 1
        return self._idf.get(term, math.log(N/1))

    def __contains__(self, term):
        return term in self._idf
