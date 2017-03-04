import numpy as np
import os,glob,logging
import torch
import PIL
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import alexnet
# import alexnet
from scipy import spatial
import tensorflow as tf
from tensorflow.python.platform import gfile
import faiss

class BaseAproxIndexer(object):

    def __init__(self,dimensions):
        self.name = "base_aprox"
        self.net = None
        self.indexed_dirs = set()
        self.index, self.files, self.findex = None, {}, 0
        self.faiss_index = faiss.IndexFlatL2(dimensions) # not approximate

    def prepare_index(self,path):
        temp_index = []
        for dirname in os.listdir(path +"/"):
            fname = "{}/{}/indexes/{}.npy".format(path,dirname,self.name)
            if dirname not in self.indexed_dirs and dirname != 'queries' and os.path.isfile(fname):
                logging.info("Starting {}".format(fname))
                self.indexed_dirs.add(dirname)
                try:
                    t = np.load(fname)
                    if max(t.shape) > 0:
                        temp_index.append(t)
                    else:
                        raise ValueError
                except:
                    logging.error("Could not load {}".format(fname))
                    pass
                else:
                    for i, f in enumerate(file(fname.replace(".npy", ".framelist")).readlines()):
                        frame_index,frame_pk = f.strip().split('_')
                        self.files[self.findex] = {
                            'frame_index':frame_index,
                            'video_primary_key':dirname,
                            'frame_primary_key':frame_pk
                        }
                        # ENGINE.store_vector(index[-1][i, :], "{}".format(findex))
                        self.findex += 1
                    logging.info("Loaded {}".format(fname))
        if self.index is None:
            self.index = np.concatenate(temp_index)
            self.index = self.index.squeeze()
            logging.info(self.index.shape)
        elif temp_index:
            self.index = np.concatenate([self.index, np.concatenate(temp_index).squeeze()])
            logging.info(self.index.shape)
        self.faiss_index.add(self.index)

    def nearest(self,image_path,n=12):
        query_vector = self.apply(image_path)
        temp = []
        dist = []
        logging.info("started query")
        return query_vector
        # ranked = np.squeeze(dist.argsort())
        # logging.info("query finished")
        # results = []
        # for i, k in enumerate(ranked[:n]):
        #     temp = {'rank':i,'algo':self.name,'dist':dist[0,k]}
        #     temp.update(self.files[k])
        #     results.append(temp)
        # return results