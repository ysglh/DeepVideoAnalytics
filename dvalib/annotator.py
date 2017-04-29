from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import os.path
import PIL

import numpy as np
import tensorflow as tf
import logging

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor

slim = tf.contrib.slim


class BaseAnnotator(object):

    def __init__(self):
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

    def __init__(self):
        super(OpenImagesAnnotator, self).__init__()
        self.name = "inception"
        self.net = None
        self.tf = True
        self.session = None
        self.graph_def = None
        self.input_image = None
        self.predictions = None
        self.num_classes = 6012
        self.top_n = 25
        self.labelmap_path = os.path.abspath(__file__).split('annotator.py')[0]+'data/2016_08/labelmap.txt'
        self.dict_path = os.path.abspath(__file__).split('annotator.py')[0]+'data/dict.csv'
        self.labelmap = [line.rstrip() for line in file(self.labelmap_path).readlines()]
        if len(self.labelmap) != self.num_classes:
            logging.error("{} lines while the number of classes is {}".format(len(self.labelmap),self.num_classes))
        self.label_dict = {}
        for line in tf.gfile.GFile(self.dict_path).readlines():
            words = [word.strip(' "\n') for word in line.split(',', 1)]
            self.label_dict[words[0]] = words[1]

    def load(self):
        if self.session is None:
            logging.warning("Loading the network {} , first apply / query will be slower".format(self.name))
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            network_path = os.path.abspath(__file__).split('annotator.py')[0]+'data/2016_08/model.ckpt'
            g = tf.Graph()
            with g.as_default():
                self.input_image = tf.placeholder(tf.string)
                processed_image = inception_preprocess(self.input_image)
                with slim.arg_scope(inception.inception_v3_arg_scope()):
                    logits, end_points = inception.inception_v3(processed_image, num_classes=self.num_classes, is_training=False)
                self.predictions = end_points['multi_predictions'] = tf.nn.sigmoid(logits, name='multi_predictions')
                saver = tf_saver.Saver()
                self.session = tf.InteractiveSession(config=config)
                saver.restore(self.session, network_path)

    def apply(self,image_path):
        self.load()
        if image_path.endswith('.png'):
            im = PIL.Image.open(image_path)
            bg = PIL.Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, im)
            image_path = image_path.replace('.png','.jpg')
            bg.save(image_path)
        img_data = tf.gfile.FastGFile(image_path).read()
        predictions_eval = np.squeeze(self.session.run(self.predictions, {self.input_image: img_data}))
        results = {self.label_dict.get(self.labelmap[idx], 'unknown'):predictions_eval[idx]
                   for idx in predictions_eval.argsort()[-self.top_n:][::-1]}
        return results



