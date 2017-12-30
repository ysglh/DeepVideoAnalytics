from base_approximator import BaseApproximator
import numpy as np
import pickle, logging
try:
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.search import LOPQSearcherLMDB
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
    from lopq.utils import compute_codes_parallel
except:
    pass


class LOPQApproximator(BaseApproximator):
    """
    An approximator converts an n-dimensional vector into PQ fine and coarse codes.
    """

    def __init__(self,name,dirname):
        super(LOPQApproximator, self).__init__()
        self.name = name
        self.dirname = dirname
        self.model = None
        self.pca_reduction = None
        self.P = None
        self.mu = None
        self.model = None
        self.permuted_inds = None
        self.model_proto_filename = "{}/model.proto".format(dirname)
        self.P_filename = self.model_proto_filename.replace('.proto', '.P.npy')
        self.entries_filename = self.model_proto_filename.replace('.proto', '.json')
        self.mu_filename = self.model_proto_filename.replace('.proto', '.mu.npy')
        self.pca_filename = self.model_proto_filename.replace('.proto', '.pca.pkl')
        self.permuted_inds_filename = self.model_proto_filename.replace('.proto', '.permind.pkl')

    def load(self):
        logging.info("Loading LOPQ indexer model {}".format(self.name))
        self.model = LOPQModel.load_proto(self.model_proto_filename)
        self.pca_reduction = pickle.load(file(self.pca_filename))
        self.P = np.load(file(self.P_filename))
        self.mu = np.load(file(self.mu_filename))
        self.permuted_inds = np.load(file(self.permuted_inds_filename))

    def approximate(self, vector):
        vector = self.get_pca_vector(vector)
        codes = self.model.predict(vector)
        return codes.coarse, codes.fine

    def get_pca_vector(self, vector):
        if self.model is None:
            self.load()
        return np.dot((self.pca_reduction.transform(vector) - self.mu), self.P).transpose().squeeze()