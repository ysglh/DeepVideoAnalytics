from dvalib.trainers import lopq_trainer
import tempfile
import numpy as np
from lopq.utils import load_xvecs


if __name__ == '__main__':
    d = tempfile.mkdtemp()
    print d
    l = lopq_trainer.LOPQTrainer(name="test",dirname=d,components=32,m=8,v=8,sub=32,source_indexer_shashum="test")
    data = load_xvecs('../..//repos/lopq/data/oxford/oxford_features.fvecs')
    print data.shape
    l.train(data)
    l.save()