import logging, json
from django.conf import settings
import celery
try:
    from dvalib import indexer, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")


from ..models import IndexEntries,QueryResults,Region,Retriever, QueryRegionResults
import io


class RetrieverTask(celery.Task):
    _visual_retriever = {}
    _retriever_object = {}
    _index_count = 0

    def get_retriever(self,retriever_pk):
        if retriever_pk not in RetrieverTask._visual_retriever:
            dr = Retriever.objects.get(pk=retriever_pk)
            RetrieverTask._retriever_object[retriever_pk] = dr
            if dr.algorithm == Retriever.EXACT:
                RetrieverTask._visual_retriever[retriever_pk] = retriever.BaseRetriever(name=dr.name)
            elif dr.algorithm == Retriever.LOPQ:
                dr.arguments['proto_filename'] = dr.proto_filename()
                RetrieverTask._visual_retriever[retriever_pk] = retriever.LOPQRetriever(name=dr.name,args=dr.arguments)
                RetrieverTask._visual_retriever[retriever_pk].load()
            else:
                raise ValueError,"{} not valid retriever algorithm".format(dr.algorithm)
        return RetrieverTask._visual_retriever[retriever_pk], RetrieverTask._retriever_object[retriever_pk]

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
                vectors, vector_entries = index_entry.load_index()
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

    def retrieve(self,event,retriever_pk,vector,count,region=None):
        index_retriever,dr = self.get_retriever(retriever_pk)
        # TODO: figure out a better way to store numpy arrays.
        if dr.algorithm == Retriever.EXACT:
            self.refresh_index(dr)
        results = index_retriever.nearest(vector=vector,n=count,retriever_pk=retriever_pk,entry_getter=entry_getter)
        # TODO: optimize this using batching
        for r in results:
            qr = QueryRegionResults() if region else QueryResults()
            if region:
                qr.query_region = region
            qr.query = event.parent_process
            qr.retrieval_event_id = event.pk
            if 'detection_primary_key' in r:
                dd = Region.objects.get(pk=r['detection_primary_key'])
                qr.detection = dd
                qr.frame_id = dd.frame_id
            else:
                qr.frame_id = r['frame_primary_key']
            qr.video_id = r['video_primary_key']
            qr.algorithm = dr.algorithm
            qr.rank = r['rank']
            qr.distance = r['dist']
            qr.save()
        event.parent_process.results_available = True
        event.parent_process.save()
        return 0

def entry_getter(kid,retriever_pk):
    return None