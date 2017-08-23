import logging
import celery
from django.conf import settings

try:
    from dvalib import analyzer
except ImportError:
    logging.warning("Could not import analyzer assuming running in front-end mode / Heroku")


class AnalyzerTask(celery.Task):
    _analyzers = None

    @property
    def get_static_analyzers(self):
        analyzers_root = "{}/analyzers/".format(settings.MEDIA_ROOT)
        if AnalyzerTask._analyzers is None:
            AnalyzerTask._analyzers = {'crnn': analyzer.CRNNAnnotator(analyzers_root+"crnn/crnn.pth")}
        return AnalyzerTask._analyzers