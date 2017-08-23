import os,logging,json
from scipy import spatial
import numpy as np
import time
from collections import namedtuple

if os.environ.get('PYTORCH_MODE',False):
    pass
elif os.environ.get('CAFFE_MODE',False):
    pass
else:
    try:
        from tensorflow.python.platform import gfile
        from facenet import facenet
        import tensorflow as tf
    except ImportError:
        logging.warning("Could not import Tensorflow assuming operating in either frontend or caffe/pytorch mode")


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
    return image_scaled, filename


def _parse_resize_vgg_function(filename):
    """
    # TODO: Verify if image channel order and mean image subtraction is done in the imported model
    :param filename:
    :return:
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=3)
    # First convert the range to 0-1 and then scale the image otherwise
    # https://github.com/tensorflow/tensorflow/issues/1763
    image_ranged= tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    image_scaled = tf.image.resize_images(image_ranged, [224, 224])
    return image_scaled, filename


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
        self.support_batching = False
        self.batch_size = 100

    def apply(self,path):
        raise NotImplementedError

    def apply_batch(self,paths):
        raise NotImplementedError

    def index_paths(self,paths):
        if self.support_batching:
            logging.info("Using batching")
            fdict = self.apply_batch(paths)
            features = [fdict[paths[i]] for i in range(len(paths))]
        else:
            features = []
            for path in paths:
                features.append(self.apply(path))
        return features


class InceptionIndexer(BaseIndexer):
    """
    Batched inception indexer
    """

    def __init__(self,model_path,batch_size=8,gpu_fraction=None,session=None):
        super(InceptionIndexer, self).__init__()
        self.name = "inception"
        self.net = None
        self.tf = True
        self.session = session
        self.network_path = model_path
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.pool3 = None
        self.filenames_placeholder = None
        self.fname = None
        self.image = None
        self.iterator = None
        self.support_batching = True
        self.batch_size = batch_size
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.2))

    def load(self):
        if self.graph_def is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            with tf.variable_scope("inception_pre"):
                self.filenames_placeholder = tf.placeholder("string",name="inception_filename")
                dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
                dataset = dataset.map(_parse_resize_inception_function)
                dataset = dataset.batch(self.batch_size)
                self.iterator = dataset.make_initializable_iterator()
            with gfile.FastGFile(self.network_path, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(self.graph_def, name='incept', input_map={'ResizeBilinear': self.image})
                self.pool3 = tf.get_default_graph().get_tensor_by_name('incept/pool_3:0')
        if self.session is None:
            logging.warning("Creating a session {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            self.session = tf.InteractiveSession(config=config)

    def apply(self,image_path):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path,]})
        f, pool3_features = self.session.run([self.fname,self.pool3])
        return np.atleast_2d(np.squeeze(pool3_features))

    def apply_batch(self,image_paths):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: image_paths})
        embeddings = {}
        batch_count = 0
        while True:
            try:
                f, emb = self.session.run([self.fname,self.pool3])
                for i,fname in enumerate(f):
                    embeddings[fname] = np.atleast_2d(np.squeeze(emb[i,:,:,:]))
                batch_count += 1
                if batch_count % 100 == 0:
                    logging.info("{} batches containing {} images indexed".format(batch_count, batch_count * self.batch_size))
            except tf.errors.OutOfRangeError:
                break
        return embeddings


class VGGIndexer(BaseIndexer):
    """
    Batched VGG indexer
    """

    def __init__(self,model_path,batch_size=8,gpu_fraction=None,session=None):
        super(VGGIndexer, self).__init__()
        self.name = "vgg"
        self.net = None
        self.tf = True
        self.model_path = model_path
        self.session = session
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0
        self.conv = None
        self.filenames_placeholder = None
        self.fname = None
        self.image = None
        self.iterator = None
        self.support_batching = True
        self.batch_size = batch_size
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.2))

    def load(self):
        if self.graph_def is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            network_path = self.model_path
            with tf.variable_scope("vgg_pre"):
                self.filenames_placeholder = tf.placeholder("string",name="vgg_filenames")
                dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
                dataset = dataset.map(_parse_resize_vgg_function)
                dataset = dataset.batch(self.batch_size)
                self.iterator = dataset.make_initializable_iterator()
            with gfile.FastGFile(network_path, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(self.graph_def, name='vgg', input_map={'images:0': self.image})
            self.conv = tf.get_default_graph().get_tensor_by_name('vgg/pool5:0')
        if self.session is None:
            logging.warning("Creating a session {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            self.session = tf.InteractiveSession(config=config)

    def apply(self,image_path):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path,]})
        f, features = self.session.run([self.fname,self.conv])
        return np.atleast_2d(np.squeeze(features).sum(axis=(0,1)))

    def apply_batch(self,image_paths):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: image_paths})
        embeddings = {}
        batch_count = 0
        while True:
            try:
                f, emb = self.session.run([self.fname,self.conv])
                for i,fname in enumerate(f):
                    embeddings[fname] = np.atleast_2d(np.squeeze(emb[i,:,:,:]).sum(axis=(0,1)))
                batch_count += 1
                if batch_count % 100 == 0:
                    logging.info("{} batches containing {} images indexed".format(batch_count, batch_count * self.batch_size))
            except tf.errors.OutOfRangeError:
                break
        return embeddings


class FacenetIndexer(BaseIndexer):

    def __init__(self,model_path,gpu_fraction=None):
        super(FacenetIndexer, self).__init__()
        self.name = "facenet"
        self.network_path = model_path
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
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.15))


    def load(self):
        if self.graph_def is None:
            logging.warning("Loading {} , first apply / query will be slower".format(self.name))
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_scale_standardize_function)
            batched_dataset = dataset.batch(self.batch_size)
            self.iterator = batched_dataset.make_initializable_iterator()
            false_phase_train = tf.constant(False)
            with gfile.FastGFile(self.network_path, 'rb') as f:
                self.graph_def = tf.GraphDef()
                self.graph_def.ParseFromString(f.read())
                self.image, self.fname = self.iterator.get_next()
                _ = tf.import_graph_def(self.graph_def, input_map={'{}:0'.format(self.input_op): self.image,'phase_train:0':false_phase_train})
                self.emb = tf.get_default_graph().get_tensor_by_name('import/{}:0'.format(self.embedding_op))
        if self.session is None:
            logging.warning("Creating a session {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            self.session = tf.InteractiveSession(config=config)

    def apply(self, image_path):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path, ]})
        f, features = self.session.run([self.fname, self.emb])
        return np.atleast_2d(np.squeeze(features))

    def apply_batch(self,image_paths):
        if self.graph_def is None or self.session is None:
            self.load()
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: image_paths})
        embeddings = {}
        batch_count = 0
        while True:
            try:
                f, emb = self.session.run([self.fname,self.emb])
                for i,fname in enumerate(f):
                    embeddings[fname] = np.atleast_2d(np.squeeze(emb[i,:,:,:]).sum(axis=(0,1)))
                batch_count += 1
                if batch_count % 100 == 0:
                    logging.info("{} batches containing {} images indexed".format(batch_count, batch_count * self.batch_size))
            except tf.errors.OutOfRangeError:
                break
        return embeddings


class BaseCustomIndexer(object):

    def __init__(self):
        self.name = "base"
        self.net = None
        self.support_batching = False
        self.batch_size = 100


    def apply(self,path):
        raise NotImplementedError

    def apply_batch(self,paths):
        raise NotImplementedError

    def index_paths(self,paths):
        batch_count = 0
        if self.support_batching:
            logging.info("Using batching")
            path_buffer = []
            fdict = {}
            for path in paths:
                path_buffer.append(path)
                if len(path_buffer) > self.batch_size:
                    fdict.update(self.apply_batch(path_buffer))
                    path_buffer = []
                    batch_count += 1
                    if batch_count % 100 == 0:
                        logging.info("{} batches containing {} images indexed".format(batch_count,batch_count*self.batch_size))
            fdict.update(self.apply_batch(path_buffer))
            features = [fdict[paths[i]] for i in range(len(paths))]
        else:
            features = []
            for path in paths:
                features.append(self.apply(path))
        return features


class CustomTFIndexer(BaseCustomIndexer):

    def __init__(self,name,network_path,input_op,embedding_op,gpu_fraction=None):
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
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.15))


    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
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