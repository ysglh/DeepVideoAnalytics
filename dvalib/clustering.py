import numpy as np
try:
    from sklearn.cross_validation import train_test_split
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
except:
    pass

class Clustering(object):

    def __init__(self,fnames,n_components,model_proto_filename):
        data = []
        self.fnames = fnames
        for fname in fnames:
            data.append(np.load(fname))
        self.data = np.concatenate(data)
        self.n_components = n_components
        self.model = None
        self.search = None
        self.pca_reduction = None
        self.P = None
        self.mu = None
        self.model_proto_filename = model_proto_filename


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
        permuted_inds = eigenvalue_allocation(2, eigenvalues)
        P = P[:, permuted_inds]
        return P, mu

    def cluster(self):
        print self.data.shape
        self.pca_reduction = PCA(n_components=self.n_components)
        self.pca_reduction.fit(self.data)
        self.data = self.pca_reduction.transform(self.data)
        print self.data.shape
        self.P, self.mu = self.pca()
        self.data = self.data - self.mu
        self.data = np.dot(self.data,self.P)
        train, test = train_test_split(self.data, test_size=0.2)
        print train.shape,test.shape
        nns = compute_all_neighbors(test, train)
        self.model = LOPQModel(V=16, M=8)
        self.model.fit(train, n_init=1)
        print "fitted"
        self.searcher = LOPQSearcher(self.model)
        print "adding data"
        self.searcher.add_data(train)
        recall, _ = get_recall(self.searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))
        self.model = LOPQModel(V=16, M=16, parameters=(self.model.Cs, None, None, None))
        self.model.fit(train, n_init=1)
        self.searcher = LOPQSearcher(self.model)
        self.searcher.add_data(train)
        recall, _ = get_recall(self.searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))
        self.model = LOPQModel(V=16, M=8, subquantizer_clusters=512, parameters=(self.model.Cs, self.model.Rs, self.model.mus, None))
        self.model.fit(train, n_init=1)
        self.searcher = LOPQSearcher(self.model)
        self.searcher.add_data(train)
        recall, _ = get_recall(self.searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))

    def save(self):
        self.model.export_proto(self.model_proto_filename)

    def load(self):
        self.mode = LOPQModel.load_proto(self.model_proto_filename)
