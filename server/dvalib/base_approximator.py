class BaseApproximator(object):
    def __init__(self):
        self.name = "base_approximate"
        self.source_model = None
        self.cloud_fs_support = False

    def load(self):
        pass

    def approximate(self, vector):
        raise NotImplementedError
