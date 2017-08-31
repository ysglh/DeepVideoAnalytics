import logging, os, json
import celery
from ..models import Detector
try:
    from dvalib import detector
except ImportError:
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")
from django.conf import settings



class DetectorTask(celery.Task):
    _detectors = {}

    @property
    def get_static_detectors(self):
        return DetectorTask._detectors

    def load_detector(self,cd):
        if cd.pk not in DetectorTask._detectors:
            if cd.detector_type == Detector.TFD:
                DetectorTask._detectors[cd.pk] = detector.TFDetector(model_path=cd.get_model_path(),
                                                                     class_index_to_string=cd.class_index_to_string)
            elif cd.detector_type == Detector.YOLO:
                DetectorTask._detectors[cd.pk] = detector.YOLODetector(cd.get_yolo_args())
            elif cd.name == 'face':
                DetectorTask._detectors[cd.pk] = detector.FaceDetector()
            elif cd.name == 'textbox':
                DetectorTask._detectors[cd.pk] = detector.TextBoxDetector()
            else:
                raise ValueError,"{}".format(cd.pk)