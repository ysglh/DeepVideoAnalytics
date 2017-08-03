import logging, json
import celery
from PIL import Image

try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from ..models import IndexEntries


class IndexerTask(celery.Task):
    _visual_indexer = None
    _clusterer = None
    _session = None

    @property
    def visual_indexer(self):
        if IndexerTask._visual_indexer is None:
            # if IndexerTask._session is None:
            #     logging.info("Creating a global shared session")
            #     config = indexer.tf.ConfigProto()
            #     config.gpu_options.per_process_gpu_memory_fraction = 0.2
            #     IndexerTask._session = indexer.tf.Session()
            IndexerTask._visual_indexer = {'inception': indexer.InceptionIndexer(),
                                           'facenet': indexer.FacenetIndexer(),
                                           'vgg': indexer.VGGIndexer()}
        return IndexerTask._visual_indexer

    def index_frames(self,media_dir, frames,visual_index,task_pk):
        visual_index.load()
        entries = []
        paths = []
        video_ids = set()
        for i, df in enumerate(frames):
            entry = {
                'frame_index': df.frame_index,
                'frame_primary_key': df.pk,
                'video_primary_key': df.video_id,
                'index': i,
                'type': 'frame'
            }
            video_ids.add(df.video_id)
            paths.append("{}/{}/frames/{}.jpg".format(media_dir, df.video_id, df.frame_index))
            entries.append(entry)
        if len(video_ids) != 1:
            raise NotImplementedError,"more/less than 1 video ids {}".format(video_ids)
        else:
            video_id = video_ids.pop()
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/frames_{}_{}.npy".format(media_dir, video_id, visual_index.name,task_pk)
        entries_fname = "{}/{}/indexes/frames_{}_{}.json".format(video_id, video_id, visual_index.name,task_pk)
        with open(feat_fname, 'w') as feats:
            np.save(feats, np.array(features))
        with open(entries_fname, 'w') as entryfile:
            json.dump(entries, entryfile)
        return visual_index.name,entries,feat_fname,entries_fname

    def index_regions(self, media_dir, regions,regions_name,visual_index):
        visual_index.load()
        video_ids = set()
        entries = []
        paths = []
        for i, d in enumerate(regions):
            entry = {
                'frame_index': d.frame.frame_index,
                'detection_primary_key': d.pk,
                'frame_primary_key': d.frame.pk,
                'video_primary_key': d.frame.video_id,
                'index': i,
                'type': d.region_type
            }
            path = "{}/{}/regions/{}.jpg".format(media_dir, d.frame.video_id, d.pk)
            video_ids.add(d.frame.video_id)
            if d.materialized:
                paths.append(path)
            else:
                img = Image.open("{}/{}/frames/{}.jpg".format(media_dir, d.frame.video_id, d.frame.frame_index))
                img2 = img.crop((d.x, d.y, d.x+d.w, d.y+d.h))
                img2.save(path)
                paths.append(path)
                d.materialized = True
                d.save()
            entries.append(entry)
        if len(video_ids) != 1:
            raise NotImplementedError,"more/less than 1 video ids {}".format(video_ids)
        else:
            video_id = video_ids.pop()
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/{}_{}.npy".format(media_dir, video_id,regions_name, visual_index.name)
        entries_fname = "{}/{}/indexes/{}_{}.json".format(media_dir, video_id,regions_name, visual_index.name)
        with open(feat_fname, 'w') as feats:
            np.save(feats, np.array(features))
        with open(entries_fname, 'w') as entryfile:
            json.dump(entries, entryfile)
        return visual_index.name,entries,feat_fname,entries_fname
