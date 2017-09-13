import logging
import celery
from django.conf import settings

try:
    from dvalib import analyzer
except ImportError:
    logging.warning("Could not import analyzer assuming running in front-end mode / Heroku")


class AnalyzerTask(celery.Task):
    _analyzers = {}

    @property
    def get_static_analyzers(self):
        return AnalyzerTask._analyzers

    def load_analyzer(self,da):
        if da.name not in AnalyzerTask._analyzers:
            aroot = "{}/analyzers/".format(settings.MEDIA_ROOT)
            if da.name == 'crnn':
                AnalyzerTask._analyzers[da.name] = analyzer.CRNNAnnotator(aroot+"{}/crnn.pth".format(da.pk))
            elif da.name == 'tagger':
                AnalyzerTask._analyzers[da.name] = analyzer.OpenImagesAnnotator(aroot+"{}/open_images.ckpt".format(da.pk))
            else:
                raise ValueError,"analyzer by id {} not found".format(da.pk)