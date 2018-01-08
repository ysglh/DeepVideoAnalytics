import logging
from django.conf import settings

try:
    from dvalib import analyzer
except ImportError:
    logging.warning("Could not import analyzer assuming running in front-end mode")


class Analyzers(object):
    _analyzers = {}

    @classmethod
    def load_analyzer(self,da):
        da.ensure()
        if da.name not in Analyzers._analyzers:
            aroot = "{}/models/".format(settings.MEDIA_ROOT)
            if da.name == 'crnn':
                Analyzers._analyzers[da.name] = analyzer.CRNNAnnotator(aroot + "{}/crnn.pth".format(da.uuid))
            elif da.name == 'tagger':
                Analyzers._analyzers[da.name] = analyzer.OpenImagesAnnotator(aroot + "{}/open_images.ckpt".format(da.uuid))
            else:
                raise ValueError,"analyzer by id {} not found".format(da.pk)