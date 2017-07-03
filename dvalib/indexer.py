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

def _parse_scale_standardize_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string,channels=3)
    # https://github.com/tensorflow/tensorflow/issues/8551
    image_scaled = tf.image.resize_image_with_crop_or_pad(image_decoded, 160, 160)
    image_standardized = tf.expand_dims(tf.image.per_image_standardization(image_scaled),0)
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


class AlexnetIndexer(BaseIndexer):

    def __init__(self):
        super(AlexnetIndexer, self).__init__()
        self.name = "alexnet"
        self.net = None
        self.transform = None
        self.index, self.files, self.findex = None, {}, 0

    def apply(self,path):
        self.load()
        tensor = self.transform(PIL.Image.open(path).convert('RGB')).unsqueeze_(0)
        if torch.cuda.is_available():
            tensor = torch.FloatTensor(tensor).cuda()
        result = self.net(Variable(tensor))
        if torch.cuda.is_available():
            return result.data.cpu().numpy()
        return result.data.numpy()


    def load(self):
        if self.net is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            self.net = alexnet(pretrained=True)
            if torch.cuda.is_available():
                self.net.cuda()
            self.transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])


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


class FacenetIndexer(BaseIndexer):

    def __init__(self):
        super(FacenetIndexer, self).__init__()
        self.name = "facenet"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.index, self.files, self.findex = None, {}, 0

    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.InteractiveSession(config=config)
            self.graph_def = tf.Graph().as_default()
            meta_file, ckpt_file, model_dir = facenet.get_model_filenames()
            self.saver = tf.train.import_meta_graph(os.path.join(model_dir, meta_file))
            self.saver.restore(self.session, os.path.join(model_dir, ckpt_file))
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.image_size = self.images_placeholder.get_shape()[1]
            self.embedding_size = self.embeddings.get_shape()[1]

    def apply(self,image_path):
        self.load()
        img = PIL.Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size,self.image_size))
        img = np.array(img)
        img = facenet.prewhiten(img)
        images = np.zeros((1, self.image_size, self.image_size, 3))
        images[0, :, :, :] = img
        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        return self.session.run(self.embeddings, feed_dict=feed_dict)

    def index_frames(self, frames, video):
        raise NotImplementedError,"Use index_faces"

    def index_faces(self,paths, paths_to_pk, output_dir, video_pk):
        self.load()
        output_dir = os.path.expanduser(output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        start_time = time.time()
        batch_size = 25
        nrof_images = len(paths)
        nrof_batches = int(np.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))
        count = 0
        path_count = {}
        entries = []
        for i in xrange(nrof_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            for eindex, fname in enumerate(paths_batch):
                count += 1
                entry = {
                    'path': fname,
                    'detection_primary_key': paths_to_pk[fname],
                    'index': eindex,
                    'type': 'detection',
                    'video_primary_key': video_pk
                }
                entries.append(entry)
            images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False, image_size=self.image_size, do_prewhiten=True)
            feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = self.session.run(self.embeddings, feed_dict=feed_dict)
        if nrof_images:
            time_avg_forward_pass = (time.time() - start_time) / float(nrof_images)
            logging.info("Forward pass took avg of %.3f[seconds/image] for %d images\n" % (
            time_avg_forward_pass, nrof_images))
            logging.info("Finally saving embeddings and gallery to: %s" % (output_dir))
        feat_fname = os.path.join(output_dir, "facenet.npy")
        entries_fname = os.path.join(output_dir, "facenet.json")
        np.save(feat_fname, emb_array)
        fh = open(entries_fname, 'w')
        json.dump(entries, fh)
        fh.close()
        return path_count, emb_array, entries, feat_fname, entries_fname


class CustomTFIndexer(BaseIndexer):

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
            dataset = dataset.map(_parse_scale_standardize_function)
            self.iterator = dataset.make_initializable_iterator()
            false_phase_train = tf.constant(False)
            with gfile.FastGFile(self.network_path, 'rb') as f:
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
