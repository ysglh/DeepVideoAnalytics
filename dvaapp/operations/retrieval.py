import logging, json
from django.conf import settings
import celery
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from ..models import IndexEntries,Clusters,QueryResults,Region,ClusterCodes
import io


class RetrieverTask(celery.Task):
    _clusterer = None
    _visual_retriever = {}
    _index_count = 0

    def get_retriever(self,dr):
        if dr.pk not in RetrieverTask._visual_retriever:
            RetrieverTask._visual_retriever[dr.pk] = retriever.BaseRetriever(name=dr.name)
        return RetrieverTask._visual_retriever[dr.pk]

    @property
    def clusterer(self):
        if RetrieverTask._clusterer is None:
            RetrieverTask._clusterer = {'inception': None,
                                      'facenet': None,
                                      'vgg':None}
        return RetrieverTask._clusterer

    def refresh_index(self, dr):
        """
        :param index_name:
        :return:
        """
        # TODO: Waiting for https://github.com/celery/celery/issues/3620 to be resolved to enabel ASYNC index updates
        # TODO improve this by either having a seperate broadcast queues or using last update timestampl
        last_count = RetrieverTask._index_count
        current_count = IndexEntries.objects.count()
        if last_count == 0 or last_count != current_count:
            # update the count
            RetrieverTask._index_count = current_count
            self.update_index(dr)

    def update_index(self,dr):
        index_entries = IndexEntries.objects.filter(**dr.source_filters)
        visual_index = RetrieverTask._visual_retriever[dr.pk]
        for index_entry in index_entries:
            if index_entry.pk not in visual_index.loaded_entries and index_entry.count > 0:
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

    def retrieve(self,iq):
        index_retriever = self.get_retriever(iq.retriever)
        index_name = iq.retriever.name
        exact = True
        results = []
        # TODO: figure out a better way to store numpy arrays.
        vector = np.load(io.BytesIO(iq.vector))
        if iq.approximate:
            if self.clusterer[index_name] is None:
                self.load_clusterer(index_name)
            if self.clusterer[index_name]:
                results = self.query_approximate(iq.count, vector, index_name)
                exact = False
        if exact:
            self.refresh_index(iq.retriever)
            results = index_retriever.nearest(vector=vector,n=iq.count)
        # TODO: optimize this using batching
        for r in results:
            qr = QueryResults()
            qr.query = iq.parent_query
            qr.indexerquery = iq
            if 'detection_primary_key' in r:
                dd = Region.objects.get(pk=r['detection_primary_key'])
                qr.detection = dd
                qr.frame_id = dd.frame_id
            else:
                qr.frame_id = r['frame_primary_key']
            qr.video_id = r['video_primary_key']
            qr.algorithm = iq.algorithm
            qr.rank = r['rank']
            qr.distance = r['dist']
            qr.save()
        iq.results = True
        iq.save()
        iq.parent_query.results_available = True
        iq.parent_query.save()
        return 0

    def query_approximate(self, n, vector, index_name):
        clusterer = self.clusterer[index_name]
        results = []
        coarse, fine, results_indexes = clusterer.apply(vector, n)
        for i, k in enumerate(results_indexes[0]):
            e = ClusterCodes.objects.get(searcher_index=k.id, clusters=clusterer.dc)
            if e.detection_id:
                results.append({
                    'rank': i + 1,
                    'dist': i,
                    'detection_primary_key': e.detection_id,
                    'frame_index': e.frame.frame_index,
                    'frame_primary_key': e.frame_id,
                    'video_primary_key': e.video_id,
                    'type': 'detection',
                })
            else:
                results.append({
                    'rank': i + 1,
                    'dist': i,
                    'frame_index': e.frame.frame_index,
                    'frame_primary_key': e.frame_id,
                    'video_primary_key': e.video_id,
                    'type': 'frame',
                })
        return results
