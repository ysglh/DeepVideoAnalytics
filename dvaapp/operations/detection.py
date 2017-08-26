import logging, os, json
import celery
try:
    from dvalib import detector
except ImportError:
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")
from django.conf import settings

cocostring = {1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus',
                              7: u'train', 8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant',
                              13: u'stop sign', 14: u'parking meter', 15: u'bench', 16: u'bird', 17: u'cat',
                              18: u'dog', 19: u'horse', 20: u'sheep', 21: u'cow', 22: u'elephant', 23: u'bear',
                              24: u'zebra', 25: u'giraffe', 27: u'backpack', 28: u'umbrella', 31: u'handbag',
                              32: u'tie', 33: u'suitcase', 34: u'frisbee', 35: u'skis', 36: u'snowboard',
                              37: u'sports ball', 38: u'kite', 39: u'baseball bat', 40: u'baseball glove',
                              41: u'skateboard', 42: u'surfboard', 43: u'tennis racket', 44: u'bottle',
                              46: u'wine glass', 47: u'cup', 48: u'fork', 49: u'knife', 50: u'spoon',
                              51: u'bowl', 52: u'banana', 53: u'apple', 54: u'sandwich', 55: u'orange',
                              56: u'broccoli', 57: u'carrot', 58: u'hot dog', 59: u'pizza', 60: u'donut',
                              61: u'cake', 62: u'chair', 63: u'couch', 64: u'potted plant', 65: u'bed',
                              67: u'dining table', 70: u'toilet', 72: u'tv', 73: u'laptop', 74: u'mouse',
                              75: u'remote', 76: u'keyboard', 77: u'cell phone', 78: u'microwave', 79: u'oven',
                              80: u'toaster', 81: u'sink', 82: u'refrigerator', 84: u'book', 85: u'clock', 86: u'vase',
                              87: u'scissors', 88: u'teddy bear', 89: u'hair drier', 90: u'toothbrush'}


class DetectorTask(celery.Task):
    _detectors = {}

    @property
    def get_static_detectors(self):
        return DetectorTask._detectors

    def load_detector(self,cd):
        if cd.pk not in DetectorTask._detectors:
            droot = "{}/detectors/".format(settings.MEDIA_ROOT)
            if cd.name == 'coco':
                DetectorTask._detectors[cd.pk] = detector.TFDetector(model_path=droot+'{}/coco_mobilenet.pb'.format(cd.pk),class_index_to_string=cocostring)
            elif cd.name == 'face':
                DetectorTask._detectors[cd.pk] = detector.FaceDetector()
            elif cd.nam == 'textbox':
                DetectorTask._detectors[cd.pk] = detector.TextBoxDetector()
            else:
                model_dir = "{}/detectors/{}/".format(settings.MEDIA_ROOT, cd.pk)
                class_names = {k: v for k, v in json.loads(cd.class_names)}
                i_class_names = {i: k for k, i in class_names.items()}
                argdict = {'root_dir':model_dir,'detector_pk':cd.pk}
                DetectorTask._detectors[cd.pk] = detector.YOLODetector(i_class_names=i_class_names,args=argdict)