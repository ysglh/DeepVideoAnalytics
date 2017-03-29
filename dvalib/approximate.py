import lopq
from lopq.search import LOPQSearcherLMDB

class ApproximateIndexer(object):

    def __init__(self,index_name,model_path,lmdb_path,V=8, M=4):
        self.model = lopq.LOPQModel(V,M)
        self.index_name = index_name
        self.searcher = None
        self.model_path = model_path
        self.lmdb_path = lmdb_path

    def load(self):
        self.model.load_proto(self.model_path)

    def prepare(self,data):
        print "fitting"
        self.model.fit(data)
        print "exporting"
        self.model.export_proto(self.model_path)
        print "starting searcher"
        self.searcher = LOPQSearcherLMDB(self.model,self.lmdb_path)
        print "adding data"
        self.add_data(data)

    def add_data(self,data):
        self.searcher.add_data(data)

    def search(self,x):
        return self.searcher.search(x,quota=100)