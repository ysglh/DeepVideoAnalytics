import logging, os
import celery
try:
    from dvalib import segmentor
except:
    logging.warning("Could not import segmentor!")

voc_class_index_to_string = {}
model_path = ""


class SegmentorTask(celery.Task):
    _segmentors = None

    @property
    def get_static_segmentors(self):
        if SegmentorTask._segmentors is None:
            SegmentorTask._segmentors = {'voc':
                                             segmentor.CRFRNNSegmentor(model_path=model_path,
                                                                       class_index_to_string=voc_class_index_to_string)}
        return SegmentorTask._segmentors