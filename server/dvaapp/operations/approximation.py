import logging, uuid, json
from django.conf import settings

try:
    from dvalib import approximator
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode")

from ..models import TrainedModel, IndexEntries


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
            model_dirname = "{}/models/{}".format(settings.MEDIA_ROOT, di.pk)
            if di.algorithm == 'LOPQ':
                Approximators._index_approximator[di.pk] = approximator.LOPQApproximator(di.name, model_dirname)
            elif di.algorithm == 'PCA':
                Approximators._index_approximator[di.pk] = approximator.PCAApproximator(di.name, model_dirname)
            else:
                raise ValueError,"unknown approximator type {}".format(di.pk)
        return Approximators._index_approximator[di.pk]

    @classmethod
    def approximate_queryset(cls,approx,da,queryset,video_id,event_id):
        new_approx_indexes = []
        for index_entry in queryset:
            uid = str(uuid.uuid1()).replace('-', '_')
            approx_ind = IndexEntries()
            vectors, entries = index_entry.load_index()
            if da.algorithm == 'LOPQ':
                for i, e in enumerate(entries):
                    e['codes'] = approx.approximate(vectors[i, :])
                entries_fname = "{}/{}/indexes/{}.json".format(settings.MEDIA_ROOT, video_id, uid)
                with open(entries_fname, 'w') as entryfile:
                    json.dump(entries, entryfile)
                approx_ind.entries_file_name = "{}.json".format(uid)
                approx_ind.features_file_name = ""
            elif da.algorithm == 'PCA':
                approx_vectors = approx.approximate(vectors)
                entries_fname = "{}/{}/indexes/{}.json".format(settings.MEDIA_ROOT, video_id, uid)
                feat_fname = "{}/{}/indexes/{}.npy".format(settings.MEDIA_ROOT, video_id, uid)
                with open(entries_fname, 'w') as entryfile:
                    json.dump(entries, entryfile)
                with open(feat_fname, 'w') as featfile:
                    np.save(featfile, approx_vectors)
                approx_ind.entries_file_name = "{}.json".format(uid)
                approx_ind.features_file_name = "{}.npy".format(uid)
            else:
                raise NotImplementedError("unknown approximation algorithm {}".format(da.algorithm))
            approx_ind.indexer_shasum = index_entry.indexer_shasum
            approx_ind.approximator_shasum = da.shasum
            approx_ind.count = index_entry.count
            approx_ind.approximate = True
            approx_ind.detection_name = index_entry.detection_name
            approx_ind.contains_detections = index_entry.contains_detections
            approx_ind.contains_frames = index_entry.contains_frames
            approx_ind.video_id = index_entry.video_id
            approx_ind.algorithm = da.name
            approx_ind.event_id = event_id
            new_approx_indexes.append(approx_ind)
        IndexEntries.objects.bulk_create(new_approx_indexes, batch_size=100)
