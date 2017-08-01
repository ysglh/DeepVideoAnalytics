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


from ..models import IndexEntries,Clusters,Video,Query,IndexerQuery,QueryResults,Region,ClusterCodes,TEvent
from collections import defaultdict
from celery.result import AsyncResult
import io



class RetrieverTask(celery.Task):
    _clusterer = None
    _visual_retriever = None

    @property
    def visual_retriever(self):
        if RetrieverTask._visual_retriever is None:
            RetrieverTask._visual_retriever = {'inception': retriever.BaseRetriever(name="inception"),
                                           'vgg': retriever.BaseRetriever(name="vgg"),
                                           'facenet': retriever.BaseRetriever(name="facenet")
                                            }
        return RetrieverTask._visual_retriever

    @property
    def clusterer(self):
        if RetrieverTask._clusterer is None:
            RetrieverTask._clusterer = {'inception': None,
                                      'facenet': None,
                                      'vgg':None}
        return RetrieverTask._clusterer

    def refresh_index(self, index_name):
        """
        # TODO: speed this up by skipping refreshes when count is unchanged.
        :param index_name:
        :return:
        """
        index_entries = IndexEntries.objects.all()
        visual_index = self.visual_retriever[index_name]
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

    def load_clusterer(self, algorithm):
        dc = Clusters.objects.all().filter(completed=True, indexer_algorithm=algorithm).last()
        if dc:
            model_file_name = "{}/clusters/{}.proto".format(settings.MEDIA_ROOT, dc.pk)
            RetrieverTask._clusterer[algorithm] = clustering.Clustering(fnames=[], m=None, v=None, sub=None,n_components=None,model_proto_filename=model_file_name, dc=dc)
            logging.warning("loading clusterer {}".format(model_file_name))
            RetrieverTask._clusterer[algorithm].load()
        else:
            logging.warning("No clusterer found switching to exact search for {}".format(algorithm))
