try:
    from lopq.search import LOPQSearcherLMDB
    from lopq import LOPQModel
    from sklearn.decomposition import PCA
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from lopq import LOPQModel, LOPQSearcher
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation


except:
    pass

def pca(data):
    """
    A simple PCA implementation that demonstrates how eigenvalue allocation
    is used to permute dimensions in order to balance the variance across
    subvectors. There are plenty of PCA implementations elsewhere. What is
    important is that the eigenvalues can be used to compute a variance-balancing
    dimension permutation.
    """
    count, D = data.shape
    mu = data.sum(axis=0) / float(count)
    summed_covar = reduce(lambda acc, x: acc + np.outer(x, x), data, np.zeros((D, D)))
    A = summed_covar / (count - 1) - np.outer(mu, mu)
    eigenvalues, P = np.linalg.eigh(A)
    permuted_inds = eigenvalue_allocation(2, eigenvalues)
    P = P[:, permuted_inds]
    return P, mu


class ApproximateIndexer(object):

    def __init__(self,index_name,model_path,lmdb_path,V=16, M=16):
        self.model = LOPQModel(V,M)
        self.index_name = index_name
        self.searcher = None
        self.model_path = model_path
        self.lmdb_path = lmdb_path

    def load(self):
        self.model.load_proto(self.model_path)

    def fit(self,train):
        print train.shape
        self.pca_reduction = PCA(n_components=256)
        self.pca_reduction.fit(train)
        train = self.pca_reduction.transform(train)
        self.P, self.mu = pca(train)
        train = np.dot(train, self.P)
        print train.shape
        self.model.fit(train, n_init=1)

    def transform(self,test):
        print test.shape
        test = self.pca_reduction.transform(test)
        test = test - self.mu
        test = np.dot(test,self.P)
        print test.shape
        return test

    def fit_model(self,train):
        self.fit(train)
        self.model.export_proto(self.model_path)
        self.searcher = LOPQSearcher(self.model) # LOPQSearcherLMDB(self.model,self.lmdb_path)

    def experiment(self,data):
        train, test = train_test_split(data, test_size=0.1)
        print data.shape,train.shape,test.shape
        nns = compute_all_neighbors(test, train)
        self.fit_model(train)
        self.searcher.add_data(self.transform(train))
        recall, _ = get_recall(self.searcher, self.transform(test), nns)
        print 'Recall (V={}, M={}, subquants={}): {}'.format(self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))

    def add_data(self,data):
        self.searcher.add_data(data)

    def search(self,x):
        return self.searcher.search(x,quota=100)