from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import sys
import os.path
from PIL import Image
import logging
import numpy as np

if os.environ.get('PYTORCH_MODE',False):
    import dvalib.crnn.utils as utils
    import dvalib.crnn.dataset as dataset
    import torch
    from torch.autograd import Variable
    import dvalib.crnn.models.crnn as crnn
    logging.info("In pytorch mode, not importing TF")
elif os.environ.get('CAFFE_MODE',False):
    pass
else:
    import tensorflow as tf
    from tensorflow.contrib.slim.python.slim.nets import inception
    from tensorflow.python.training import saver as tf_saver
    slim = tf.contrib.slim


class BaseAnnotator(object):

    def __init__(self):
        self.label_set = None
        pass

    def apply(self,image_path):
        pass


def inception_preprocess(image, central_fraction=0.875):
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
    # image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.expand_dims(image, [0])
    # TODO try tf.image.resize_image_with_crop_or_pad and tf.image.extract_glimpse
    image = tf.image.resize_bilinear(image, [299, 299], align_corners=False)
    # Center the image about 128.0 (which is done during training) and normalize.
    image = tf.multiply(image, 1.0 / 127.5)
    return tf.subtract(image, 1.0)


class OpenImagesAnnotator(BaseAnnotator):

    def __init__(self,model_path,gpu_fraction=None):
        super(OpenImagesAnnotator, self).__init__()
        self.name = "inception"
        self.object_name = "tag"
        self.net = None
        self.tf = True
        self.session = None
        self.label_set = 'open_images_tags'
        self.graph_def = None
        self.input_image = None
        self.predictions = None
        self.num_classes = 6012
        self.top_n = 25
        self.network_path = model_path
        self.labelmap_path = model_path.replace('open_images.ckpt','open_images_labelmap.txt')
        self.dict_path = model_path.replace('open_images.ckpt','open_images_dict.csv')
        self.labelmap = [line.rstrip() for line in file(self.labelmap_path).readlines()]
        if gpu_fraction:
            self.gpu_fraction = gpu_fraction
        else:
            self.gpu_fraction = float(os.environ.get('GPU_MEMORY', 0.15))

    def load(self):
        if self.session is None:
            if len(self.labelmap) != self.num_classes:
                logging.error("{} lines while the number of classes is {}".format(len(self.labelmap), self.num_classes))
            self.label_dict = {}
            for line in tf.gfile.GFile(self.dict_path).readlines():
                words = [word.strip(' "\n') for word in line.split(',', 1)]
                self.label_dict[words[0]] = words[1]
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
            g = tf.Graph()
            with g.as_default():
                self.input_image = tf.placeholder(tf.string)
                processed_image = inception_preprocess(self.input_image)
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    logits, end_points = inception.inception_v3(processed_image, num_classes=self.num_classes, is_training=False)
                self.predictions = end_points['multi_predictions'] = tf.nn.sigmoid(logits, name='multi_predictions')
                saver = tf_saver.Saver()
                self.session = tf.InteractiveSession(config=config)
                saver.restore(self.session, self.network_path)

    def apply(self,image_path):
        if self.session is None:
            self.load()
        img_data = tf.gfile.FastGFile(image_path).read()
        predictions_eval = np.squeeze(self.session.run(self.predictions, {self.input_image: img_data}))
        results = {self.label_dict.get(self.labelmap[idx], 'unknown'):predictions_eval[idx]
                   for idx in predictions_eval.argsort()[-self.top_n:][::-1]}
        labels = [t for t,v in results.iteritems() if v > 0.1]
        text = " ".join(labels)
        metadata = {t:round(100.0*v,2) for t,v in results.iteritems() if v > 0.1}
        return self.object_name,text,metadata,labels


class CRNNAnnotator(BaseAnnotator):

    def __init__(self,model_path):
        super(CRNNAnnotator, self).__init__()
        self.session = None
        self.object_name = "text"
        self.model_path = model_path
        self.alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.cuda = False

    def load(self):
        logging.info("Loding CRNN model first apply will be slow")
        if torch.cuda.is_available():
            self.session = crnn.CRNN(32, 1, 37, 256, 1).cuda()
            self.cuda = True
        else:
            self.session = crnn.CRNN(32, 1, 37, 256, 1)
        self.session.load_state_dict(torch.load(self.model_path))
        self.session.eval()
        self.converter = utils.strLabelConverter(self.alphabet)
        self.transformer = dataset.resizeNormalize((100, 32))

    def apply(self,image_path):
        if self.session is None:
            self.load()
        image = Image.open(image_path).convert('L')
        if self.cuda:
            image = self.transformer(image).cuda()
        else:
            image = self.transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)
        preds = self.session(image)
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return self.object_name,sim_pred,{},None


