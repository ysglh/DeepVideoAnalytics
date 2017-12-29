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
    from lopq.search import LOPQSearcherLMDB
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
    from lopq.utils import compute_codes_parallel
except:
    pass


IndexRange = namedtuple('IndexRange',['start','end'])


class BaseRetriever(object):

    def __init__(self,name):
        self.name = name
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

    def nearest(self, vector=None, n=12,retriever_pk=None,entry_getter=None):
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

    def __init__(self,name,args,test_mode=False):
        super(BaseRetriever, self).__init__()
        self.name = name
        self.loaded_entries = {}
        self.index, self.files, self.findex = None, {}, 0
        self.support_batching = False

    def load(self):
        self.model = LOPQModel.load_proto(self.model_proto_filename)
        self.pca_reduction = pickle.load(file(self.pca_filename))
        self.P = np.load(file(self.P_filename))
        self.mu = np.load(file(self.mu_filename))
        self.permuted_inds = np.load(file(self.permuted_inds_filename))
        self.searcher = LOPQSearcherLMDB(model=self.model,lmdb_path=self.model_lmdb_filename)

    def apply(self,vector,count=None):
        vector = np.dot((self.pca_reduction.transform(vector) - self.mu), self.P).transpose().squeeze()
        codes = self.model.predict(vector)
        if count:
            results = self.searcher.search(vector,quota=count)
        else:
            results = None
        return codes.coarse,codes.fine,results

    def nearest(self,vector=None, n=12,retriever_pk=None,entry_getter=None):
        results = []
        coarse, fine, results_indexes = self.apply(vector, n)
        for i, k in enumerate(results_indexes[0]):
            e = entry_getter(k.id,retriever_pk)
            if e.detection_id:
                results.append({
                    'rank': i + 1,
                    'dist': i,
                    'detection_primary_key': e.detection_id,
                    'frame_index': e.frame.frame_index,
                    'frame_primary_key': e.frame_id,
                    'video_primary_key': e.video_id,
                    'type': 'detection',
                })
            else:
                results.append({
                    'rank': i + 1,
                    'dist': i,
                    'frame_index': e.frame.frame_index,
                    'frame_primary_key': e.frame_id,
                    'video_primary_key': e.video_id,
                    'type': 'frame',
                })
        return results