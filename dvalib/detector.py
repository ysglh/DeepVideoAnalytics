import os,logging,subprocess, time
import tensorflow as tf
import PIL
from dvalib.ssd.nets import ssd_vgg_300, np_methods
from dvalib.ssd.preprocessing import ssd_vgg_preprocessing
from collections import defaultdict
import numpy as np
from scipy import misc
from collections import defaultdict
from .facenet import facenet
from .facenet.align import detect_face
import random
from time import sleep
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def pil_to_array(pilImage):
    """
    Load a PIL image and return it as a numpy array.  For grayscale
    images, the return array is MxN.  For RGB images, the return value
    is MxNx3.  For RGBA images the return value is MxNx4
    """
    def toarray(im, dtype=np.uint8):
        """Return a 1D array of dtype."""
        # Pillow wants us to use "tobytes"
        if hasattr(im, 'tobytes'):
            x_str = im.tobytes('raw', im.mode)
        else:
            x_str = im.tostring('raw', im.mode)
        x = np.fromstring(x_str, dtype)
        return x

    if pilImage.mode in ('RGBA', 'RGBX'):
        im = pilImage  # no need to convert images
    elif pilImage.mode == 'L':
        im = pilImage  # no need to luminance images
        # return MxN luminance array
        x = toarray(im)
        x.shape = im.size[1], im.size[0]
        return x
    elif pilImage.mode == 'RGB':
        # return MxNx3 RGB array
        im = pilImage  # no need to RGB images
        x = toarray(im)
        x.shape = im.size[1], im.size[0], 3
        return x
    elif pilImage.mode.startswith('I;16'):
        # return MxN luminance array of uint16
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>u2')
        else:
            x = toarray(im, '<u2')
        x.shape = im.size[1], im.size[0]
        return x.astype('=u2')
    else:  # try to convert to an rgba image
        try:
            im = pilImage.convert('RGBA')
        except ValueError:
            raise RuntimeError('Unknown image mode')

    # return MxNx4 RGBA array
    x = toarray(im)
    x.shape = im.size[1], im.size[0], 4
    return x


class BaseDetector(object):

    def __init__(self):
        pass

    def detect(self,path):
        pass

    def load(self):
        pass


class TFDetector(BaseDetector):

    def __init__(self,model_path,class_index_to_string):
        super(TFDetector, self).__init__()
        self.model_path = model_path
        self.class_index_to_string = class_index_to_string

    def detect(self,image_path,min_score=0.1):
        plimg = PIL.Image.open(image_path).convert('RGB')
        img = pil_to_array(plimg)
        image_np_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                 feed_dict={self.image_tensor: image_np_expanded})
        detections = []
        for i, _ in enumerate(boxes[0]):
            if scores[0][i] > min_score:
                detections.append({
                    'box': boxes[0][i],
                    'score': scores[0][i],
                    'object_name': self.class_index_to_string[int(classes[0][i])]
                })
        return detections

    def load(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


class SSDetector(BaseDetector):

    def __init__(self):
        self.isess = None
        self.name = "SSD"
        self.classnames =  {v[0]:k for k,v in VOC_LABELS.iteritems()}
        logging.info("Detector created")

    def load(self):
        slim = tf.contrib.slim
        net_shape = (300, 300)
        data_format = 'NHWC' # NCHW not defined
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(self.img_input, None, None,net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)
        self.ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(self.ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=None) # ask paul about reuse = None
        network_path = os.path.abspath(__file__).split('detector.py')[0] + 'ssd/checkpoints/ssd_300_vgg.ckpt'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.isess = tf.InteractiveSession(config=config)
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, network_path)
        self.ssd_anchors = self.ssd_net.anchors(net_shape)

    def detect(self,wframes, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        detections = defaultdict(list)
        if self.isess is None:
            self.load()
        for wf in wframes:
            plimg = PIL.Image.open(wf.local_path()).convert('RGB')
            img = pil_to_array(plimg)
            rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],feed_dict={self.img_input: img})
            rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, self.ssd_anchors,select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
            rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
            shape = img.shape
            for i,bbox in enumerate(rbboxes):
                top,left = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
                bot,right = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
                detections[wf.primary_key].append({
                    'top':top,
                    'bot':bot,
                    'left':left,
                    'right':right,
                    'confidence':100*rscores[i],
                    'name':"{}_{}".format(self.name,self.classnames[rclasses[i]])})
        return detections


class FaceDetector():

    def __init__(self):
        self.image_size = 182
        self.margin = 44
        self.gpu_memory_fraction = 0.2

    def detect(self,wframes):
        sleep(random.random())
        logging.info('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        random_key = np.random.randint(0, high=99999)
        aligned_paths = defaultdict(list)
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for frame in wframes:
            image_path = frame.local_path()
            nrof_images_total += 1
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                logging.info(errorMessage)
            else:
                if img.ndim < 2:
                    logging.info('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    det_all = bounding_boxes[:, 0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    for boxindex in range(nrof_faces):
                        det = np.squeeze(det_all[boxindex, :])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - self.margin / 2, 0)
                        bb[1] = np.maximum(det[1] - self.margin / 2, 0)
                        bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                        nrof_successfully_aligned += 1
                        aligned_paths[image_path].append((scaled,bb))
        logging.info('Total number of images: %d' % nrof_images_total)
        logging.info('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        return aligned_paths
