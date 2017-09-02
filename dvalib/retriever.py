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

    def __init__(self,name,args):
        data = []
        self.name = name
        self.fnames = args.get('fnames',[])
        self.entries = []
        for fname in self.fnames:
            nmat = np.load(fname)
            if nmat.ndim > 2:
                nmat = nmat.squeeze()
            data.append(nmat)
            for e in json.load(file(fname.replace('npy','json'))):
                self.entries.append(e)
        if data:
            if len(data) > 1:
                self.data = np.concatenate(data)
            else:
                self.data = data[0]
            logging.info(self.data.shape)
        self.test_mode = args.get('test_mode',False)
        self.n_components = int(args['components'])
        self.m = int(args['m'])
        self.v = int(args['v'])
        self.sub = int(args['sub'])
        self.model = None
        self.searcher = None
        self.pca_reduction = None
        self.P = None
        self.mu = None
        self.model_proto_filename = args['proto_filename']
        self.P_filename = args['proto_filename'].replace('.proto','.P.npy')
        self.mu_filename = args['proto_filename'].replace('.proto','.mu.npy')
        self.pca_filename = args['proto_filename'].replace('.proto', '.pca.pkl')
        self.model_lmdb_filename = args['proto_filename'].replace('.proto', '_lmdb')
        self.permuted_inds_filename = args['proto_filename'].replace('.proto', '.permuted_inds.pkl')
        self.permuted_inds = None

    def pca(self):
        """
        A simple PCA implementation that demonstrates how eigenvalue allocation
        is used to permute dimensions in order to balance the variance across
        subvectors. There are plenty of PCA implementations elsewhere. What is
        important is that the eigenvalues can be used to compute a variance-balancing
        dimension permutation.
        """
        count, D = self.data.shape
        mu = self.data.sum(axis=0) / float(count)
        summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), self.data, np.zeros((D, D)))
        A = summed_covar / (count - 1) - np.outer(mu, mu)
        eigenvalues, P = np.linalg.eigh(A)
        self.permuted_inds = eigenvalue_allocation(2, eigenvalues)
        P = P[:, self.permuted_inds]
        return P, mu

    def cluster(self):
        self.pca_reduction = PCA(n_components=self.n_components)
        self.pca_reduction.fit(self.data)
        self.data = self.pca_reduction.transform(self.data)
        self.P, self.mu = self.pca()
        self.data = self.data - self.mu
        self.data = np.dot(self.data,self.P)
        # train, test = train_test_split(self.data, test_size=0.2)
        self.model = LOPQModel(V=self.v, M=self.m, subquantizer_clusters=self.sub)
        self.model.fit(self.data, n_init=1) # replace self.data by train
        for i,e in enumerate(self.entries): # avoid doing this twice again in searcher
            r = self.model.predict(self.data[i])
            e['coarse'] = r.coarse
            e['fine'] = r.fine
            e['index'] = i
        self.searcher = LOPQSearcherLMDB(self.model,self.model_lmdb_filename)
        # if self.test_mode:
        #     self.searcher.add_data(train)
        #     nns = compute_all_neighbors(test, train)
        #     recall, _ = get_recall(self.searcher, test, nns)
        #     print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))
        # else:
        self.searcher.add_data(self.data)

    def find(self):
        i,selected = random.choice([k for k in enumerate(self.entries)])
        print selected
        for k in self.searcher.get_result_quota(self.data[i],10):
            print k

    def save(self):
        with open(self.model_proto_filename,'w') as f:
            self.model.export_proto(f)
            with open(self.pca_filename,'w') as out:
                pickle.dump(self.pca_reduction,out)
            with open(self.P_filename, 'w') as out:
                np.save(out,self.P)
            with open(self.mu_filename, 'w') as out:
                np.save(out,self.mu)
            with open(self.permuted_inds_filename, 'w') as out:
                pickle.dump(self.permuted_inds,out)
            self.searcher.env.close()

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