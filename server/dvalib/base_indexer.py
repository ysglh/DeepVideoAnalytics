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
