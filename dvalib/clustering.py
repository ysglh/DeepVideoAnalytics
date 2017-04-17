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

    def __init__(self,fname,n_components):
        self.data = np.load(fname)
        self.n_components = n_components

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
        pca_reduction = PCA(n_components=32)
        pca_reduction.fit(self.data)
        self.data = pca_reduction.transform(self.data)
        print self.data.shape
        P, mu = self.pca()
        self.data = self.data - mu
        data = np.dot(self.data,P)
        train, test = train_test_split(self.data, test_size=0.2)
        print train.shape,test.shape
        nns = compute_all_neighbors(test, train)
        m = LOPQModel(V=16, M=8)
        m.fit(train, n_init=1)
        print "fitted"
        searcher = LOPQSearcher(m)
        print "adding data"
        searcher.add_data(train)
        recall, _ = get_recall(searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m.V, m.M, m.subquantizer_clusters, str(recall))
        m2 = LOPQModel(V=16, M=16, parameters=(m.Cs, None, None, None))
        m2.fit(train, n_init=1)
        searcher = LOPQSearcher(m2)
        searcher.add_data(train)
        recall, _ = get_recall(searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m2.V, m2.M, m2.subquantizer_clusters, str(recall))
        m3 = LOPQModel(V=16, M=8, subquantizer_clusters=512, parameters=(m.Cs, m.Rs, m.mus, None))
        m3.fit(train, n_init=1)
        searcher = LOPQSearcher(m3)
        searcher.add_data(train)
        recall, _ = get_recall(searcher, test, nns)
        print 'Recall (V=%d, M=%d, subquants=%d): %s' % (m3.V, m3.M, m3.subquantizer_clusters, str(recall))