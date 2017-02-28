import numpy as np
import os,glob,logging
import torch
import PIL
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import alexnet
import tensorflow as tf
from tensorflow.python.platform import gfile
from dvalib.ssd.nets import ssd_vgg_300, ssd_common, np_methods
from dvalib.ssd.preprocessing import ssd_vgg_preprocessing
import matplotlib.image as mpimg


class BaseDetector(object):

    def __init__(self):
        pass

    def detect(self,path):
        pass

    def load(self):
        pass


class SSDetector(BaseDetector):

    def __init__(self):
        self.isess = None
        self.name = "SSD"

    def load(self):
        slim = tf.contrib.slim
        net_shape = (300, 300)
        data_format = 'NHWC' # NCHW not defined
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None,net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=None) # ask paul about reuse = None
        network_path = os.path.abspath(__file__).split('detector.py')[0] + 'ssd/checkpoints/ssd_300_vgg.ckpt'
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, network_path)
        self.ssd_anchors = ssd_net.anchors(net_shape)

    def detect(self,path, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        img = mpimg.imread(path)
        if self.isess is None:
            logging.warning("Loading the SSD network")
            self.load()
            logging.warning("Loading finished")
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],feed_dict={self.img_input: img})
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors,select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes