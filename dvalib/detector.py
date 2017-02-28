import os,logging,subprocess
import tensorflow as tf
import PIL
from dvalib.ssd.nets import ssd_vgg_300, np_methods
from dvalib.ssd.preprocessing import ssd_vgg_preprocessing
from collections import defaultdict
import numpy as np

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


class YOLODetector(object):

    def __init__(self):
        self.name = "YOLO9000"

    def detect(self,wframes):
        darknet_path = os.path.join(os.path.abspath(__file__).split('detector.py')[0],'../darknet/')
        list_path = "{}/{}_list.txt".format(darknet_path, os.getpid())
        output_path = "{}/{}_output.txt".format(darknet_path, os.getpid())
        logging.info(darknet_path)
        path_to_pk = {}
        with open(list_path, 'w') as framelist:
            for frame in wframes:
                framelist.write('{}\n'.format(frame.local_path()))
                path_to_pk[frame.local_path()] = frame.primary_key
        # ./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights data/list.txt
        with open(output_path, 'w') as output:
            args = ["./darknet", 'detector', 'test', 'cfg/combine9k.data', 'cfg/yolo9000.cfg', 'yolo9000.weights',list_path]
            logging.info(args)
            returncode = subprocess.call(args, cwd=darknet_path, stdout=output)
        if returncode == 0:
            detections = defaultdict(list)
            for line in file(output_path):
                if line.strip():
                    temp = {}
                    frame_path, name, confidence, left, right, top, bot = line.strip().split('\t')
                    if frame_path not in path_to_pk:
                        raise ValueError, frame_path
                    temp['top'] = int(top)
                    temp['left'] = int(left)
                    temp['right'] = int(right)
                    temp['bot'] = int(bot)
                    temp['confidence'] = 100.0*float(confidence)
                    temp['name'] = "{}_{}".format(self.name,name.replace(' ', '_'))
                    detections[path_to_pk[frame_path]].append(temp)
            return detections
        else:
            raise ValueError,returncode


class SSDetector(BaseDetector):

    def __init__(self):
        self.isess = None
        self.name = "SSD"
        self.classnames =  {v[0]:k for k,v in VOC_LABELS.iteritems()}

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

    def detect(self,wframes, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        detections = defaultdict(list)
        if self.isess is None:
            logging.warning("Loading the SSD network")
            self.load()
            logging.warning("Loading finished")
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

if 'YOLO_ENABLE' in os.environ:
    DETECTORS = {'ssd': SSDetector(),'yolo':YOLODetector()}
else:
    DETECTORS = {'ssd':SSDetector() }
