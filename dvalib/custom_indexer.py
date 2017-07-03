import numpy as np
import os,logging,json
import PIL
try:
    import torch
    from torch.autograd import Variable
    from torchvision import transforms
    from torchvision.models import alexnet
except ImportError:
    logging.warning("Could not import torch")
from scipy import spatial
try:
    from tensorflow.python.platform import gfile
    from facenet import facenet
    import tensorflow as tf
except ImportError:
    logging.warning("Could not import Tensorflow assuming operating in either frontend or caffe/pytorch mode")
import time
from collections import namedtuple


IndexRange = namedtuple('IndexRange',['start','end'])


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string,channels=3)
    return image_decoded, filename


class BaseCustomIndexer(object):

    def __init__(self):
        self.name = "base"
        self.net = None
        self.loaded_entries = {}
        self.index, self.files, self.findex = None, {}, 0
        self.support_batching = False
        self.batch_size = 100

    def load_index(self,numpy_matrix,entries):
        temp_index = [numpy_matrix, ]
        for i, e in enumerate(entries):
            self.files[self.findex] = e
            self.findex += 1
        if self.index is None:
            self.index = np.concatenate(temp_index)
            self.index = self.index.squeeze()
            logging.info(self.index.shape)
        else:
            self.index = np.concatenate([self.index, np.concatenate(temp_index).squeeze()])
            logging.info(self.index.shape)

    def nearest(self, image_path, n=12, query_vector=None):
        if query_vector is None:
            query_vector = self.apply(image_path)
        temp = []
        dist = []
        for k in xrange(self.index.shape[0]):
            temp.append(self.index[k])
            if (k+1) % 50000 == 0:
                temp = np.transpose(np.dstack(temp)[0])
                dist.append(spatial.distance.cdist(query_vector,temp))
                temp = []
        if temp:
            temp = np.transpose(np.dstack(temp)[0])
            dist.append(spatial.distance.cdist(query_vector,temp))
        dist = np.hstack(dist)
        ranked = np.squeeze(dist.argsort())
        results = []
        for i, k in enumerate(ranked[:n]):
            temp = {'rank':i+1,'algo':self.name,'dist':float(dist[0,k])}
            temp.update(self.files[k])
            results.append(temp)
        return results # Next also return computed query_vector

    def apply(self,path):
        raise NotImplementedError

    def apply_batch(self,paths):
        raise NotImplementedError

    def index_paths(self,paths):
        if self.support_batching:
            logging.info("Using batching")
            path_buffer = []
            fdict = {}
            for path in paths:
                path_buffer.append(path)
                if len(path_buffer) > self.batch_size:
                    fdict.update(self.apply_batch(path_buffer))
                    path_buffer = []
            fdict.update(self.apply_batch(path_buffer))
            features = [fdict[paths[i]] for i in range(len(paths))]
        else:
            features = []
            for path in paths:
                features.append(self.apply(path))
        return features


class CustomTFIndexer(BaseCustomIndexer):

    def __init__(self,name,network_path,input_op,embedding_op):
        super(CustomTFIndexer, self).__init__()
        self.name = name
        self.network_path = network_path
        self.embedding_op = embedding_op
        self.input_op = input_op
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.image = None
        self.filenames_placeholder = None
        self.emb = None

    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.InteractiveSession(config=config)
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_function)
            self.iterator = dataset.make_initializable_iterator()
            false_phase_train = tf.constant(False)
            with gfile.FastGFile(self.network_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(graph_def, input_map={'{}:0'.format(self.input_op): self.image})
                self.emb = self.session.graph.get_tensor_by_name('import/{}:0'.format(self.embedding_op))


    def apply(self, image_path):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path, ]})
        f, features = self.session.run([self.fname, self.emb])
        return np.atleast_2d(np.squeeze(features))