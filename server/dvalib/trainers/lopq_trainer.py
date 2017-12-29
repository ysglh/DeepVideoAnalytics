import pickle, json
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
        self.model_proto_filename = "{}/model.proto".format(dirname)
        self.P_filename = self.model_proto_filename.replace('.proto', '.P.npy')
        self.mu_filename = self.model_proto_filename.replace('.proto', '.mu.npy')
        self.pca_filename = self.model_proto_filename.replace('.proto', '.pca.pkl')
        self.permuted_inds_filename = self.model_proto_filename.replace('.proto', '.permind.pkl')

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
        with open(self.model_proto_filename, 'w') as f:
            self.model.export_proto(f)
            with open(self.pca_filename, 'w') as out:
                pickle.dump(self.pca_reduction, out)
            with open(self.P_filename, 'w') as out:
                np.save(out, self.P)
            with open(self.mu_filename, 'w') as out:
                np.save(out, self.mu)
            with open(self.permuted_inds_filename, 'w') as out:
                pickle.dump(self.permuted_inds, out)