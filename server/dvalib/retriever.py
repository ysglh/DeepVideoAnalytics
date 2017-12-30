import numpy as np
import logging
from scipy import spatial
from collections import namedtuple
import pickle
import json
import random
import logging
try:
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
    from lopq.utils import compute_codes_parallel
except:
    pass


IndexRange = namedtuple('IndexRange',['start','end'])


class BaseRetriever(object):

    def __init__(self,name):
        self.name = name
        self.approximate = False
        self.net = None
        self.loaded_entries = {}
        self.index, self.files, self.findex = None, {}, 0
        self.support_batching = False

    def load_index(self,numpy_matrix,entries):
        temp_index = [numpy_matrix, ]
        for i, e in enumerate(entries):
            self.files[self.findex] = e
            self.findex += 1
        if self.index is None:
            self.index = np.atleast_2d(np.concatenate(temp_index).squeeze())
            logging.info(self.index.shape)
        else:
            self.index = np.concatenate([self.index, np.atleast_2d(np.concatenate(temp_index).squeeze())])
            logging.info(self.index.shape)

    def nearest(self, vector=None, n=12):
        dist = None
        results = []
        if self.index is not None:
            # logging.info("{} and {}".format(vector.shape,self.index.shape))
            dist = spatial.distance.cdist(vector,self.index)
        if dist is not None:
            ranked = np.squeeze(dist.argsort())
            for i, k in enumerate(ranked[:n]):
                temp = {'rank':i+1,'algo':self.name,'dist':float(dist[0,k])}
                temp.update(self.files[k])
                results.append(temp)
        return results # Next also return computed query_vector


class LOPQRetriever(BaseRetriever):

    def __init__(self,name,approximator):
        super(BaseRetriever, self).__init__()
        self.approximate = True
        self.name = name
        self.loaded_entries = []
        self.entries = []
        self.support_batching = False
        self.approximator = approximator
        self.approximator.load()
        self.searcher = LOPQSearcher(model=self.approximator.model)

    def load_index(self,numpy_matrix=None,entries=None):
        codes = []
        ids = []
        last_index = len(self.entries)
        for i, e in enumerate(entries):
            codes.append(e['codes'])
            ids.append(i+last_index)
            self.entries.append(e)
        self.searcher.add_codes(codes,ids)

    def nearest(self,vector=None,n=12):
        results = []
        pca_vec = self.approximator.get_pca_vector(vector)
        results_indexes, visited = self.searcher.search(pca_vec,quota=n)
        for r in results_indexes:
            results.append(self.entries[r.id])
        return results