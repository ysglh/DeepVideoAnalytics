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

    def __init__(self,index_name,model_path,lmdb_path,V=16, M=8):
        self.model = LOPQModel(V,M)
        self.index_name = index_name
        self.searcher = None
        self.model_path = model_path
        self.lmdb_path = lmdb_path

    def load(self):
        self.model.load_proto(self.model_path)


    def prepare(self,data):
        print data.shape
        train, test = train_test_split(data, test_size=0.2)
        nns = compute_all_neighbors(test, train)
        pca_reduction = PCA(n_components=32)
        pca_reduction.fit(train)
        train = pca_reduction.transform(train)
        print train.shape
        P, mu = pca(train)
        train = train - mu
        train = np.dot(train,P)
        test = pca_reduction.transform(test)
        print test.shape
        test = test - mu
        test = np.dot(test,P)
        print "fitting"
        self.model.fit(train,n_init=1)
        print "exporting"
        self.model.export_proto(self.model_path)
        # print "starting searcher"
        # self.searcher = LOPQSearcherLMDB(self.model,self.lmdb_path)
        # print "adding data"
        # self.add_data(data)
        searcher = LOPQSearcher(self.model)
        print "adding data"
        searcher.add_data(train)
        recall, _ = get_recall(searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (self.model.V, self.model.M, self.model.subquantizer_clusters, str(recall))


    def add_data(self,data):
        self.searcher.add_data(data)

    def search(self,x):
        return self.searcher.search(x,quota=100)