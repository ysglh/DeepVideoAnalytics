import sys,os
sys.path.append(os.path.dirname(__file__))
from dvalib import external_indexed

try:
    import numpy as np
    from sklearn.cross_validation import train_test_split
    from sklearn.decomposition import PCA
    from lopq import LOPQModel, LOPQSearcher
    from lopq.eval import compute_all_neighbors, get_recall
    from lopq.model import eigenvalue_allocation
except:
    pass


def load_oxford_data():
    return data


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

def main(input_dir='/Users/aub3/temptest/gtin/',output_dir="/Users/aub3/temptest/products"):
        products = external_indexed.ProductsIndex(path=output_dir)
        # products.prepare(input_dir)
        products.build_approximate()
        data = products.data
        # data = load_oxford_data()
        print data.shape
        pca_reduction = PCA(n_components=32)
        pca_reduction.fit(data)
        data = pca_reduction.transform(data)
        print data.shape
        P, mu = pca(data)
        data = data - mu
        data = np.dot(data,P)
        train, test = train_test_split(data, test_size=0.2)
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


if __name__ == '__main__':
    main()