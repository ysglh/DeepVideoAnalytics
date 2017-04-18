import numpy as np
import pickle
import json
try:
    from sklearn.cross_validation import train_test_split
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.search import LOPQSearcherLMDB
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
    from lopq.utils import compute_codes_parallel
except:
    pass

class Clustering(object):

    def __init__(self,fnames,n_components,model_proto_filename,test_mode=False):
        data = []
        self.fnames = fnames
        self.entries = []
        for fname in fnames:
            data.append(np.load(fname))
            for e in json.load(file(fname.replace('npy','json'))):
                self.entries.append(e)
        self.data = np.concatenate(data)
        self.test_mode = test_mode
        self.n_components = n_components
        self.model = None
        self.search = None
        self.pca_reduction = None
        self.P = None
        self.mu = None
        self.model_proto_filename = model_proto_filename
        self.searcher_lmdb_filename = model_proto_filename.replace('.proto','.lmdb')
        self.P_filename = model_proto_filename.replace('.proto','.P.npy')
        self.mu_filename = model_proto_filename.replace('.proto','.mu.npy')
        self.pca_filename = model_proto_filename.replace('.proto', '.pca.pkl')
        self.permuted_inds_filename = model_proto_filename.replace('.proto', '.permuted_inds.pkl')
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
        train, test = train_test_split(self.data, test_size=0.2)
        self.model = LOPQModel(V=16, M=16)
        self.model.fit(train, n_init=1)
        self.searcher = LOPQSearcherLMDB(self.model,self.searcher_lmdb_filename)

        if self.test_mode:
            self.searcher.add_data(train)
            nns = compute_all_neighbors(test, train)
            recall, _ = get_recall(self.searcher, test, nns)
            print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))
        else:
            self.searcher.add_data(self.data)
        return compute_codes_parallel(self.data,self.model)

    def save(self):
        self.model.export_proto(self.model_proto_filename)
        self.searcher.env.close()
        with open(self.pca_filename,'w') as out:
            pickle.dump(self.pca_reduction,out)
        with open(self.P_filename, 'w') as out:
            np.save(out,self.P)
        with open(self.mu_filename, 'w') as out:
            np.save(out,self.mu)
        with open(self.permuted_inds_filename, 'w') as out:
            pickle.dump(self.permuted_inds,out)

    def load(self):
        self.model = LOPQModel.load_proto(self.model_proto_filename)
