import logging
from .approximation import Approximators
try:
    from dvalib import indexer, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode")


from ..models import IndexEntries,QueryResults,Region,Retriever, QueryRegionResults


class Retrievers(object):
    _visual_retriever = {}
    _retriever_object = {}
    _index_count = 0

    @classmethod
    def get_retriever(cls,retriever_pk):
        if retriever_pk not in cls._visual_retriever:
            dr = Retriever.objects.get(pk=retriever_pk)
            cls._retriever_object[retriever_pk] = dr
            if dr.algorithm == Retriever.EXACT and dr.approximator_shasum.strip():
                approximator, da = Approximators.get_approximator_by_shasum(dr.approximator_shasum)
                da.ensure()
                approximator.load()
                cls._visual_retriever[retriever_pk] = retriever.BaseRetriever(name=dr.name,approximator=approximator)
            elif dr.algorithm == Retriever.EXACT:
                cls._visual_retriever[retriever_pk] = retriever.BaseRetriever(name=dr.name)
            elif dr.algorithm == Retriever.LOPQ:
                approximator, da = Approximators.get_approximator_by_shasum(dr.approximator_shasum)
                da.ensure()
                approximator.load()
                cls._visual_retriever[retriever_pk] = retriever.LOPQRetriever(name=dr.name,
                                                                              approximator=approximator)

            else:
                raise ValueError,"{} not valid retriever algorithm".format(dr.algorithm)
        return cls._visual_retriever[retriever_pk], cls._retriever_object[retriever_pk]

    @classmethod
    def refresh_index(cls, dr):
        """
        :param index_name:
        :return:
        """
        # This has a BUG where total count of index entries remains unchanged
        # TODO: Waiting for https://github.com/celery/celery/issues/3620 to be resolved to enabel ASYNC index updates
        # TODO improve this by either having a seperate broadcast queues or using last update timestampl
        last_count = cls._index_count
        current_count = IndexEntries.objects.count()
        visual_index = cls._visual_retriever[dr.pk]
        if last_count == 0 or last_count != current_count or len(visual_index.loaded_entries) == 0:
            # update the count
            cls._index_count = current_count
            cls.update_index(dr)

    @classmethod
    def update_index(cls,dr):
        source_filters = dr.source_filters.copy()
        if dr.indexer_shasum:
            source_filters['indexer_shasum'] = dr.indexer_shasum
        if dr.approximator_shasum:
            source_filters['approximator_shasum'] = dr.approximator_shasum
        else:
            source_filters['approximator_shasum'] = None # Required otherwise approximate index entries are selected
        index_entries = IndexEntries.objects.filter(**source_filters)
        visual_index = cls._visual_retriever[dr.pk]
        for index_entry in index_entries:
            if index_entry.pk not in visual_index.loaded_entries and index_entry.count > 0:
                vectors, entries = index_entry.load_index()
                if visual_index.algorithm == "LOPQ":
                    logging.info("loading approximate index {}".format(index_entry.pk))
                    start_index = len(visual_index.entries)
                    visual_index.load_index(entries=entries)
                    visual_index.loaded_entries[index_entry.pk] = indexer.IndexRange(start=start_index,
                                                                                     end=len(visual_index.entries)-1)
                else:
                    logging.info("Starting {} in {} with shape {}".format(index_entry.video_id, visual_index.name,
                                                                          vectors.shape))
                    try:
                        start_index = visual_index.findex
                        visual_index.load_index(vectors, entries)
                        visual_index.loaded_entries[index_entry.pk] = indexer.IndexRange(start=start_index,
                                                                                         end=visual_index.findex-1)
                    except:
                        logging.info("ERROR Failed to load {} vectors shape {} entries {}".format(
                            index_entry.video_id,vectors.shape,len(entries)))
                    else:
                        logging.info("finished {} in {}, current shape {}, range".format(index_entry.video_id,
                                                                             visual_index.name,
                                                                             visual_index.index.shape,
                                                                             visual_index.loaded_entries[
                                                                                 index_entry.pk].start,
                                                                             visual_index.loaded_entries[
                                                                                 index_entry.pk].end,
                                                                             ))

    @classmethod
    def retrieve(cls,event,retriever_pk,vector,count,region=None):
        index_retriever,dr = cls.get_retriever(retriever_pk)
        cls.refresh_index(dr)
        # TODO: figure out a better way to store numpy arrays
        results = index_retriever.nearest(vector=vector,n=count)
        # TODO: optimize this using batching
        for rank,r in enumerate(results):
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
            qr.rank = r.get('rank',rank)
            qr.distance = r.get('dist',rank)
            qr.save()
        event.parent_process.results_available = True
        event.parent_process.save()
        return 0