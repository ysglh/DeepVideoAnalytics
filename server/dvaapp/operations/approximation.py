import logging
from django.conf import settings

try:
    from dvalib import approximator
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode")

from ..models import TrainedModel


class Approximators(object):
    _index_approximator = {}
    _name_to_index = {}
    _shasum_to_index = {}
    _session = None

    @classmethod
    def get_approximator_by_name(cls,name):
        if name not in Approximators._name_to_index:
            di = TrainedModel.objects.get(name=name,model_type=TrainedModel.APPROXIMATOR)
            Approximators._name_to_index[name] = di
        else:
            di = Approximators._name_to_index[name]
        return cls.get_approximator(di),di

    @classmethod
    def get_approximator_by_shasum(cls,shasum):
        if shasum not in Approximators._shasum_to_index:
            di = TrainedModel.objects.get(shasum=shasum,model_type=TrainedModel.APPROXIMATOR)
            Approximators._shasum_to_index[shasum] = di
        else:
            di = Approximators._shasum_to_index[shasum]
        return cls.get_approximator(di),di

    @classmethod
    def get_approximator_by_pk(cls,pk):
        di = TrainedModel.objects.get(pk=pk)
        if di.model_type != TrainedModel.APPROXIMATOR:
            raise ValueError("Model {} id: {} is not an Indexer".format(di.name,di.pk))
        return cls.get_approximator(di),di
    
    @classmethod
    def get_approximator(cls,di):
        di.ensure()
        if di.pk not in Approximators._index_approximator:
            iroot = "{}/models/".format(settings.MEDIA_ROOT)
            if di.algorithm == 'LOPQ':
                Approximators._index_approximator[di.pk] = approximator.LOPQApproximator(di.name,
                                                                                     iroot + "{}/".format(di.pk))
            else:
                raise ValueError,"unknown approximator type {}".format(di.pk)
        return Approximators._index_approximator[di.pk]