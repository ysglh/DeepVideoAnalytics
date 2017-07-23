import logging, os
import celery
try:
    from dvalib import analyzer
except ImportError:
    logging.warning("Could not import analyzer assuming running in front-end mode / Heroku")


class AnalyzerTask(celery.Task):
    _detectors = None
    _clusterer = None

    @property
    def get_static_detectors(self):
        if AnalyzerTask._analyzers is None:
            AnalyzerTask._analyzers = {'tag': analyzer.OpenImagesAnnotator(),
                                       'text': analyzer.CRNNAnnotator()}
        return AnalyzerTask._analyzers