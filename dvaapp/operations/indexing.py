import logging, json, uuid
import celery
from PIL import Image
from django.conf import settings

try:
    from dvalib import indexer, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from ..models import IndexEntries, Indexer


class IndexerTask(celery.Task):
    _visual_indexer = {}
    _name_to_index = {}
    _session = None

    def get_index_by_name(self,name):
        if name not in IndexerTask._name_to_index:
            di = Indexer.objects.get(name=name)
            IndexerTask._name_to_index[name] = di
        else:
            di = IndexerTask._name_to_index[name]
        return self.get_index(di),di

    def get_index(self,di):
        if di.pk not in IndexerTask._visual_indexer:
            iroot = "{}/indexers/".format(settings.MEDIA_ROOT)
            if di.name == 'inception':
                IndexerTask._visual_indexer[di.pk] = indexer.InceptionIndexer(iroot+"{}/network.pb".format(di.pk))
            elif di.name == 'facenet':
                IndexerTask._visual_indexer[di.pk] = indexer.FacenetIndexer(iroot+"{}/facenet.pb".format(di.pk))
            elif di.name == 'vgg':
                IndexerTask._visual_indexer[di.pk] = indexer.VGGIndexer(iroot+"{}/vgg.pb".format(di.pk))
            else:
                raise ValueError,"unregistered indexer with id {}".format(di.pk)
        return IndexerTask._visual_indexer[di.pk]

    def index_queryset(self,di,visual_index,event,target,queryset):
        visual_index.load()
        entries, paths, images = [], [], {}
        for i, df in enumerate(queryset):
            if target == 'frames':
                entry = {'frame_index': df.frame_index,
                         'frame_primary_key': df.pk,
                         'video_primary_key': event.video_id,
                         'index': i,
                         'type': 'frame'}
            elif target == 'regions':
                entry = {
                    'frame_index': df.frame.frame_index,
                    'detection_primary_key': df.pk,
                    'frame_primary_key': df.frame.pk,
                    'video_primary_key': event.video_id,
                    'index': i,
                    'type': df.region_type
                }
                if not df.materialized:
                    frame_path = df.frame_path()
                    if frame_path not in images:
                        images[frame_path] = Image.open(frame_path)
                    img2 = images[frame_path].crop((df.x, df.y, df.x + df.w, df.y + df.h))
                    img2.save(df.path())
            else:
                raise ValueError,"{} target not configured".format(target)
            paths.append(df.path())
            entries.append(entry)
        if entries:
            features = visual_index.index_paths(paths)
            uid = str(uuid.uuid1()).replace('-','_')
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
            i.source_id = event.pk
            i.source_filter_json = event.arguments
            i.save()
