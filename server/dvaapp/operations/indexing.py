import logging, json, uuid, tempfile, os
from PIL import Image
from django.conf import settings

try:
    from dvalib import indexer, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode")

from ..models import IndexEntries, TrainedModel


class Indexers(object):
    _visual_indexer = {}
    _name_to_index = {}
    _session = None

    @classmethod
    def get_index_by_name(cls,name):
        if name not in Indexers._name_to_index:
            di = TrainedModel.objects.get(name=name,model_type=TrainedModel.INDEXER)
            Indexers._name_to_index[name] = di
        else:
            di = Indexers._name_to_index[name]
        return cls.get_index(di),di
    
    @classmethod
    def get_index_by_pk(cls,pk):
        di = TrainedModel.objects.get(pk=pk)
        if di.model_type != TrainedModel.INDEXER:
            raise ValueError("Model {} id: {} is not an Indexer".format(di.name,di.pk))
        return cls.get_index(di),di
    
    @classmethod
    def get_index(cls,di):
        di.ensure()
        if di.pk not in Indexers._visual_indexer:
            iroot = "{}/models/".format(settings.MEDIA_ROOT)
            if di.name == 'inception':
                Indexers._visual_indexer[di.pk] = indexer.InceptionIndexer(iroot + "{}/network.pb".format(di.uuid))
            elif di.name == 'facenet':
                Indexers._visual_indexer[di.pk] = indexer.FacenetIndexer(iroot + "{}/facenet.pb".format(di.uuid))
            elif di.algorithm == 'vgg':
                Indexers._visual_indexer[di.pk] = indexer.VGGIndexer(iroot + "{}/{}".format(di.uuid,di.files[0]['filename']))
            else:
                raise ValueError,"unregistered indexer with id {}".format(di.pk)
        return Indexers._visual_indexer[di.pk]

    @classmethod
    def index_queryset(cls,di,visual_index,event,target,queryset, cloud_paths=False):
        visual_index.load()
        temp_root = tempfile.mkdtemp()
        entries, paths, images = [], [], {}
        for i, df in enumerate(queryset):
            if target == 'frames':
                entry = {'frame_index': df.frame_index,
                         'frame_primary_key': df.pk,
                         'video_primary_key': event.video_id,
                         'index': i,
                         'type': 'frame'}
                if cloud_paths:
                    paths.append(df.path('{}://{}'.format(settings.CLOUD_FS_PREFIX,settings.MEDIA_BUCKET)))
                else:
                    paths.append(df.path())
            elif target == 'regions':
                entry = {
                    'frame_index': df.frame.frame_index,
                    'detection_primary_key': df.pk,
                    'frame_primary_key': df.frame.pk,
                    'video_primary_key': event.video_id,
                    'index': i,
                    'type': df.region_type
                }
                if df.full_frame:
                    paths.append(df.frame_path())
                elif df.materialized:
                    paths.append(df.path())
                else:
                    frame_path = df.frame_path()
                    if frame_path not in images:
                        images[frame_path] = Image.open(frame_path)
                    img2 = images[frame_path].crop((df.x, df.y, df.x + df.w, df.y + df.h))
                    region_path = df.path(temp_root=temp_root)
                    img2.save(region_path)
                    paths.append(region_path)
            else:
                raise ValueError,"{} target not configured".format(target)
            entries.append(entry)
        if entries:
            logging.info(paths)  # adding temporary logging to check whether s3:// paths are being correctly used.
            # TODO Ensure that "full frame"/"regions" are not repeatedly indexed.
            features = visual_index.index_paths(paths)
            uid = str(uuid.uuid1()).replace('-','_')
            dirnames = ['{}/{}/'.format(settings.MEDIA_ROOT,event.video_id),
                        '{}/{}/indexes/'.format(settings.MEDIA_ROOT,event.video_id)]
            for dirname in dirnames:
                if not os.path.isdir(dirname):
                    try:
                        os.mkdir(dirname)
                    except:
                        logging.exception("error creating {}".format(dirname))
                        pass
            feat_fname = "{}/{}/indexes/{}.npy".format(settings.MEDIA_ROOT,event.video_id,uid)
            entries_fname = "{}/{}/indexes/{}.json".format(settings.MEDIA_ROOT,event.video_id,uid)
            with open(feat_fname, 'w') as feats:
                np.save(feats, np.array(features))
            with open(entries_fname, 'w') as entryfile:
                json.dump(entries, entryfile)
            i = IndexEntries()
            i.video_id = event.video_id
            i.count = len(entries)
            i.contains_detections = target == "regions"
            i.contains_frames = target == "frames"
            i.detection_name = '{}_subset_by_{}'.format(target,event.pk)
            i.algorithm = di.name
            i.indexer = di
            i.indexer_shasum = di.shasum
            i.entries_file_name = entries_fname.split('/')[-1]
            i.features_file_name = feat_fname.split('/')[-1]
            i.event_id = event.pk
            i.source_filter_json = event.arguments
            i.save()
