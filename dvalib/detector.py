import os,logging,subprocess, time
import tensorflow as tf
import PIL
import numpy as np
from scipy import misc
from collections import defaultdict
from .facenet import facenet
from .facenet.align import detect_face
import random
from time import sleep


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string,channels=3)
    return tf.expand_dims(image_decoded, 0), filename



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
        self.session = None
        self.dataset = None
        self.filenames_placeholder = None
        self.image = None
        self.fname = None

    def detect(self,image_path,min_score=0.20):
        self.session.run(self.iterator.initializer, feed_dict={self.filenames_placeholder: [image_path,]})
        (fname, boxes, scores, classes, num_detections) = self.session.run(
            [self.fname,self.boxes, self.scores, self.classes, self.num_detections])
        detections = []
        for i, _ in enumerate(boxes[0]):
            plimg = PIL.Image.open(image_path)
            frame_width, frame_height = plimg.size
            shape = (frame_height,frame_width)
            if scores[0][i] > min_score:
                top,left = (int(boxes[0][i][0] * shape[0]), int(boxes[0][i][1] * shape[1]))
                bot,right = (int(boxes[0][i][2] * shape[0]), int(boxes[0][i][3] * shape[1]))
                detections.append({
                    'x': left,
                    'y':top,
                    'w':right-left,
                    'h':bot-top,
                    'score': scores[0][i],
                    'object_name': self.class_index_to_string[int(classes[0][i])]
                })
        return detections

    def load(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.filenames_placeholder = tf.placeholder("string")
            dataset = tf.contrib.data.Dataset.from_tensor_slices(self.filenames_placeholder)
            dataset = dataset.map(_parse_function)
            self.iterator = dataset.make_initializable_iterator()
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                self.image, self.fname = self.iterator.get_next()
                tf.import_graph_def(self.od_graph_def, name='',input_map={'image_tensor': self.image})
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.15
            self.session = tf.Session(graph=self.detection_graph,config=config)
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


class FaceDetector():

    def __init__(self,session=None):
        self.image_size = 182
        self.margin = 44
        self.gpu_memory_fraction = 0.1
        self.session = session
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

    def load(self):
        logging.info('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with self.session.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.session, None)

    def detect(self,image_path):
        aligned = []
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            logging.info(errorMessage)
        else:
            if img.ndim < 2:
                logging.info('Unable to align "%s"' % image_path)
                return []
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
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
                    left, top, right, bottom = bb[0], bb[1], bb[2], bb[3]
                    aligned.append({
                        'scaled':scaled,
                        'x': left,
                        'y':top,
                        'w':right-left,
                        'h':bottom-top,
                    })
            return aligned