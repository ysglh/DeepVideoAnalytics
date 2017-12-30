import pickle, hashlib
import numpy as np

try:
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
    from lopq.utils import compute_codes_parallel
except:
    pass


class LOPQTrainer(object):

    def __init__(self,name,components,m,v,sub,dirname,source_indexer_shashum):
        self.name = name
        self.n_components = int(components)
        self.m = int(m)
        self.v = int(v)
        self.dirname = dirname
        self.sub = int(sub)
        self.model = None
        self.pca_reduction = None
        self.P = None
        self.mu = None
        self.permuted_inds = None
        self.source_indexer_shashum = source_indexer_shashum

    def pca(self,training_data):
        """
        A simple PCA implementation that demonstrates how eigenvalue allocation
        is used to permute dimensions in order to balance the variance across
        sub vectors. There are plenty of PCA implementations elsewhere. What is
        important is that the eigenvalues can be used to compute a variance-balancing
        dimension permutation.
        """
        count, D = training_data.shape
        mu = training_data.sum(axis=0) / float(count)
        summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), training_data, np.zeros((D, D)))
        A = summed_covar / (count - 1) - np.outer(mu, mu)
        eigenvalues, P = np.linalg.eigh(A)
        self.permuted_inds = eigenvalue_allocation(2, eigenvalues)
        P = P[:, self.permuted_inds]
        return P, mu

    def train(self,training_data):
        self.pca_reduction = PCA(n_components=self.n_components)
        self.pca_reduction.fit(training_data)
        training_data = self.pca_reduction.transform(training_data)
        self.P, self.mu = self.pca(training_data)
        training_data = training_data - self.mu
        training_data = np.dot(training_data, self.P)
        self.model = LOPQModel(V=self.v, M=self.m, subquantizer_clusters=self.sub)
        self.model.fit(training_data, n_init=1)  # replace self.data by train

    def save(self):
        model_proto_filename = "{}/model.proto".format(self.dirname)
        P_filename = "{}/model.P.npy".format(self.dirname)
        mu_filename = "{}/model.mu.npy".format(self.dirname)
        pca_filename = "{}/model.pca.pkl".format(self.dirname)
        permind_filename = "{}/model.permind.pkl".format(self.dirname)
        with open(model_proto_filename, 'w') as f:
            self.model.export_proto(f)
        with open(pca_filename, 'w') as out:
            pickle.dump(self.pca_reduction, out)
        with open(P_filename, 'w') as out:
            np.save(out, self.P)
        with open(mu_filename, 'w') as out:
            np.save(out, self.mu)
        with open(permind_filename, 'w') as out:
            pickle.dump(self.permuted_inds, out)
        j = {"name":self.name,
              "algorithm":"LOPQ",
              "shasum":hashlib.sha1(file(model_proto_filename).read()).hexdigest(),
              "model_type":"P",
               "arguments":{
                   'm':self.m,
                   'v': self.v,
                   'sub': self.sub,
                   'components': self.n_components,
                   'indexer_shasum': self.source_indexer_shashum
               },
               "files": [
                   {"filename":"model.proto","url":"{}/model.proto".format(self.dirname)},
                   {"filename":"model.P.npy","url":"{}/model.P.npy".format(self.dirname)},
                   {"filename":"model.mu.npy","url":"{}/model.mu.npy".format(self.dirname)},
                   {"filename":"model.pca.pkl","url":"{}/model.pca.pkl".format(self.dirname)},
                   {"filename":"model.permind.pkl","url":"{}/model.permind.pkl".format(self.dirname)}
               ]
            }
        return j