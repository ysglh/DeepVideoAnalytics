import logging

class BaseIndexer(object):
    def __init__(self):
        self.name = "base"
        self.net = None
        self.support_batching = False
        self.batch_size = 100
        self.num_parallel_calls = 3
        self.cloud_fs_support = False

    def apply(self, path):
        raise NotImplementedError

    def apply_batch(self, paths):
        raise NotImplementedError

    def index_paths(self, paths):
        if self.support_batching:
            logging.info("Using batching")
            fdict = self.apply_batch(paths)
            features = [fdict[paths[i]] for i in range(len(paths))]
        else:
            features = []
            for path in paths:
                features.append(self.apply(path))
        return features


class BaseApproximateIndexer(object):
    def __init__(self):
        self.name = "base_approximate"
        self.source_model = None
        self.cloud_fs_support = False

    def apply(self, path):
        if self.source_model is None:
            raise ValueError("Source model is not available")
        else:
             return self.approximate(self.source_model.apply(path))

    def apply_batch(self, paths):
        if self.source_model.support_batching:
            vecs = self.source_model.apply_batch(paths)
        else:
            vecs = [self.source_model.apply(p) for p in paths]
        return self.index_vectors(vecs)

    def index_paths(self, paths):
        features = []
        for path in paths:
            features.append(self.apply(path))
        return features

    def approximate(self, vector):
        raise NotImplementedError

    def index_vectors(self, vectors):
        features = []
        for v in vectors:
            features.append(self.approximate(v))
        return features