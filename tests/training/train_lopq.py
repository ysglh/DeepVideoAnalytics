import os
from dvalib.trainers import lopq_trainer
import numpy as np


if __name__ == '__main__':
    l = lopq_trainer.LOPQTrainer(name="Facenet LOPQ trained on LFW",
                                 dirname=os.path.join(os.path.dirname('__file__'),"facenet_lopq/"),
                                 components=32,m=16,v=16,sub=128,
                                 source_indexer_shashum="9f99caccbc75dcee8cb0a55a0551d7c5cb8a6836")
    data = np.load('facenet.npy')
    print data.shape
    l.train(data)
    l.save()