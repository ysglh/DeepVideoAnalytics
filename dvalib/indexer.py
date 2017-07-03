import numpy as np
import os,logging,json
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


def _parse_resize_inception_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=3)
    # Cannot use decode_image but decode_png decodes both jpeg as well as png
    # https://github.com/tensorflow/tensorflow/issues/8551
    image_scaled = tf.image.resize_images(image_decoded, [299, 299])
    image_standardized = tf.image.per_image_standardization(image_scaled)
    return image_standardized, filename

def _parse_scale_standardize_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=3)
    # Cannot use decode_image but decode_png decodes both jpeg as well as png
    # https://github.com/tensorflow/tensorflow/issues/8551
    image_scaled = tf.image.resize_images(image_decoded, [160, 160])
    image_standardized = tf.image.per_image_standardization(image_scaled)
    return image_standardized, filename


class BaseIndexer(object):

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


class InceptionIndexer(BaseIndexer):

    def __init__(self):
        super(InceptionIndexer, self).__init__()
        self.name = "inception"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.pool3 = None
        self.filenames_placeholder = None
        self.fname = None
        self.image = None
        self.iterator = None
        self.support_batching = True


    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.InteractiveSession(config=config)
            network_path = os.path.abspath(__file__).split('indexer.py')[0]+'data/network.pb'
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_function)
            self.iterator = dataset.make_initializable_iterator()
            with gfile.FastGFile(network_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(graph_def, name='incept', input_map={'DecodeJpeg': self.image})
                self.pool3 = self.session.graph.get_tensor_by_name('incept/pool_3:0')

    def apply(self,image_path):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path,]})
        f, pool3_features = self.session.run([self.fname,self.pool3])
        return np.atleast_2d(np.squeeze(pool3_features))

    def apply_batch(self,image_paths):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: image_paths})
        embeddings = {}
        while True:
            try:
                f, emb = self.session.run([self.fname,self.pool3])
                embeddings[f] = np.atleast_2d(np.squeeze(emb))
            except tf.errors.OutOfRangeError:
                break
        return embeddings


class BInceptionIndexer(BaseIndexer):
    """
    Batched inception indexer
    """

    def __init__(self):
        super(BInceptionIndexer, self).__init__()
        self.name = "batchedinception"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.pool3 = None
        self.filenames_placeholder = None
        self.fname = None
        self.image = None
        self.iterator = None
        self.support_batching = True
        self.batch_size = 64

    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.InteractiveSession(config=config)
            network_path = os.path.abspath(__file__).split('indexer.py')[0]+'data/network.pb'
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_resize_inception_function)
            dataset = dataset.batch(self.batch_size)
            self.iterator = dataset.make_initializable_iterator()
            with gfile.FastGFile(network_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(graph_def, name='incept', input_map={'ResizeBilinear': self.image})
                self.pool3 = self.session.graph.get_tensor_by_name('incept/pool_3:0')

    def apply(self,image_path):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path,]})
        f, pool3_features = self.session.run([self.fname,self.pool3])
        return np.atleast_2d(np.squeeze(pool3_features))

    def apply_batch(self,image_paths):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: image_paths})
        embeddings = {}
        while True:
            try:
                f, emb = self.session.run([self.fname,self.pool3])
                print emb.shape
                for i,fname in enumerate(f):
                    embeddings[fname] = np.atleast_2d(np.squeeze(emb[i,:,:,:]))
            except tf.errors.OutOfRangeError:
                break
        return embeddings


class FacenetIndexer(BaseIndexer):

    def __init__(self):
        super(FacenetIndexer, self).__init__()
        self.name = "facenet"
        self.network_path = os.path.abspath(__file__).split('indexer.py')[0]+'data/facenet.pb'
        self.embedding_op = "embeddings"
        self.input_op = "input"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.image = None
        self.filenames_placeholder = None
        self.emb = None
        self.batch_size = 32

    def load(self):
        if self.session is None:
            logging.warning("Loading {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.InteractiveSession(config=config)
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_scale_standardize_function)
            batched_dataset = dataset.batch(self.batch_size)
            self.iterator = batched_dataset.make_initializable_iterator()
            false_phase_train = tf.constant(False)
            with gfile.FastGFile(self.network_path, 'rb') as f:
                print self.network_path
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(graph_def, input_map={'{}:0'.format(self.input_op): self.image,'phase_train:0':false_phase_train})
                self.emb = self.session.graph.get_tensor_by_name('import/{}:0'.format(self.embedding_op))

    def apply(self, image_path):
        if self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path, ]})
        f, features = self.session.run([self.fname, self.emb])
        return np.atleast_2d(np.squeeze(features))

    def index_faces(self, paths, paths_to_pk, output_dir, video_pk):
        self.load()
        entries = []
        output_dir = os.path.expanduser(output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        path_count = 0
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder:paths})
        embeddings = []
        while True:
            try:
                fnames, emb_array = self.session.run([self.fname, self.emb])
                embeddings.append(emb_array)
                # print len(fnames),emb_array.shape
                for eindex, fname in enumerate(fnames):
                    entry = {
                        'path': fname,
                        'detection_primary_key': paths_to_pk[fname],
                        'index': path_count,
                        'type': 'detection',
                        'video_primary_key': video_pk
                    }
                    path_count += 1
                    entries.append(entry)
            except tf.errors.OutOfRangeError:
                break
        feat_fname = os.path.join(output_dir, "facenet.npy")
        entries_fname = os.path.join(output_dir, "facenet.json")
        if embeddings:
            embeddings = np.squeeze(np.vstack(embeddings))
            # print embeddings.shape
            np.save(feat_fname, embeddings)
            fh = open(entries_fname, 'w')
            json.dump(entries, fh)
            fh.close()
        else:
            embeddings = None
        return path_count, embeddings, entries, feat_fname, entries_fname
