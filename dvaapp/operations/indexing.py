import logging, json, base64
import boto3
from django.conf import settings
import celery
from dva.celery import app
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

    @property
    def clusterer(self):
        if IndexerTask._clusterer is None:
            IndexerTask._clusterer = {'inception': None,
                                      'facenet': None,
                                      'vgg':None}
        return IndexerTask._clusterer

    def refresh_index(self, index_name):
        """
        # TODO: speed this up by skipping refreshes when count is unchanged.
        :param index_name:
        :return:
        """
        index_entries = IndexEntries.objects.all()
        visual_index = self.visual_indexer[index_name]
        for index_entry in index_entries:
            if index_entry.pk not in visual_index.loaded_entries and index_entry.algorithm == index_name and index_entry.count > 0:
                fname = "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id,
                                                  index_entry.features_file_name)
                vectors = indexer.np.load(fname)
                vector_entries = json.load(file("{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id,
                                                                          index_entry.entries_file_name)))
                logging.info("Starting {} in {} with shape {}".format(index_entry.video_id, visual_index.name,vectors.shape))
                start_index = visual_index.findex
                try:
                    visual_index.load_index(vectors, vector_entries)
                except:
                    logging.info("ERROR Failed to load {} vectors shape {} entries {}".format(index_entry.video_id,vectors.shape,len(vector_entries)))
                visual_index.loaded_entries[index_entry.pk] = indexer.IndexRange(start=start_index,
                                                                                 end=visual_index.findex - 1)
                logging.info("finished {} in {}, current shape {}, range".format(index_entry.video_id,
                                                                                 visual_index.name,
                                                                                 visual_index.index.shape,
                                                                                 visual_index.loaded_entries[
                                                                                     index_entry.pk].start,
                                                                                 visual_index.loaded_entries[
                                                                                     index_entry.pk].end,
                                                                                 ))