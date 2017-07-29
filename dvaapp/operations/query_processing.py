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

    def load_clusterer(self, algorithm):
        dc = Clusters.objects.all().filter(completed=True, indexer_algorithm=algorithm).last()
        if dc:
            model_file_name = "{}/clusters/{}.proto".format(settings.MEDIA_ROOT, dc.pk)
            IndexerTask._clusterer[algorithm] = clustering.Clustering(fnames=[], m=None, v=None, sub=None,n_components=None,model_proto_filename=model_file_name, dc=dc)
            logging.warning("loading clusterer {}".format(model_file_name))
            IndexerTask._clusterer[algorithm].load()
        else:
            logging.warning("No clusterer found switching to exact search for {}".format(algorithm))


class RetrieverTask(celery.Task):
    _clusterer = None
    _visual_retriever =  None

    @property
    def visual_retriever(self):
        if IndexerTask._visual_retriever is None:
            IndexerTask._visual_retriever = {'inception': retriever.BaseRetriever(name="inception"),
                                           'vgg': retriever.BaseRetriever(name="vgg"),
                                           'facenet': retriever.BaseRetriever(name="facenet")
                                            }
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
            IndexerTask._clusterer[algorithm] = clustering.Clustering(fnames=[], m=None, v=None, sub=None,n_components=None,model_proto_filename=model_file_name, dc=dc)
            logging.warning("loading clusterer {}".format(model_file_name))
            IndexerTask._clusterer[algorithm].load()
        else:
            logging.warning("No clusterer found switching to exact search for {}".format(algorithm))


def query_approximate(local_path, n, visual_index, clusterer):
    vector = visual_index.apply(local_path)
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


class QueryProcessing(object):

    def __init__(self):
        self.query = None
        self.media_dir = None
        self.indexer_queries = []
        self.task_results = {}
        self.context = {}
        self.dv = None
        self.visual_indexes = settings.VISUAL_INDEXES

    def store_and_create_video_object(self):
        self.dv = Video()
        self.dv.name = 'query_{}'.format(self.query.pk)
        self.dv.dataset = True
        self.dv.query = True
        self.dv.parent_query = self.query
        self.dv.save()
        if settings.HEROKU_DEPLOY:
            query_key = "queries/{}.png".format(self.query.pk)
            query_frame_key = "{}/frames/0.png".format(self.dv.pk)
            s3 = boto3.resource('s3')
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_key, Body=self.query.image_data)
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_frame_key, Body=self.query.image_data)
        else:
            query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.query.pk)
            with open(query_path, 'w') as fh:
                fh.write(self.query.image_data)

    def create_from_request(self, request):
        count = request.POST.get('count')
        excluded_index_entries_pk = json.loads(request.POST.get('excluded_index_entries'))
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
        approximate = True if request.POST.get('approximate') == 'true' else False
        image_data_url = request.POST.get('image_url')
        user = request.user if request.user.is_authenticated else None
        self.query = Query()
        self.query.approximate = approximate
        if not (user is None):
            self.query.user = user
        image_data = base64.decodestring(image_data_url[22:])
        self.query.image_data = image_data
        self.query.save()
        self.store_and_create_video_object()
        for k in selected_indexers:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k
            iq.count = count
            if excluded_index_entries_pk:
                # !!fix this only the indexer specific
                iq.excluded_index_entries_pk = [int(epk) for epk in excluded_index_entries_pk]
            iq.approximate = approximate
            iq.save()
            self.indexer_queries.append(iq)
        return self.query

    def create_from_json(self, j, user=None):
        """
        Create query from JSON
        {
        'image_data':base64.encodestring(file('tests/query.png').read()),
        'indexers':[
            {
                'algorithm':'facenet',
                'count':10,
                'approximate':False
            }
            ]
        }
        :param j: JSON encoded query
        :param user:
        :return:
        """
        self.query = Query()
        if not (user is None):
            self.query.user = user
        if j['image_data_b64'].strip():
            image_data = base64.decodestring(j['image_data_b64'])
            self.query.image_data = image_data
        self.query.save()
        self.store_and_create_video_object()
        for k in j['indexers']:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k['algorithm']
            iq.count = k['count']
            iq.excluded_index_entries_pk = k['excluded_index_entries_pk'] if 'excluded_index_entries_pk' in k else []
            iq.approximate = k['approximate']
            iq.save()
            self.indexer_queries.append(iq)
        return self.query

    def send_tasks(self):
        for iq in self.indexer_queries:
            task_name = 'perform_indexing'
            queue_name = self.visual_indexes[iq.algorithm]['indexer_queue']
            jargs = json.dumps({
                'iq_id':iq.pk,
                'index':iq.algorithm,
                'target':'query',
                'next_tasks':[
                    { 'task_name': 'perform_retrieval',
                      'arguments': {'iq_id': iq.pk,'index':iq.algorithm}
                     }
                ]
            })
            next_task = TEvent.objects.create(video=self.dv, operation=task_name, arguments_json=jargs)
            self.task_results[iq.algorithm] = app.send_task(task_name, args=[next_task.pk, ], queue=queue_name, priority=5)
            self.context[iq.algorithm] = []

    def wait(self,timeout=120):
        for visual_index_name, result in self.task_results.iteritems():
            try:
                logging.info("Waiting for {}".format(visual_index_name))
                _ = result.get(timeout=timeout)
            except Exception, e:
                raise ValueError(e)

    def collect_results(self):
        self.context = defaultdict(list)
        for r in QueryResults.objects.all().filter(query=self.query):
            self.context[r.algorithm].append((r.rank,
                                         {'url': '{}{}/regions/{}.jpg'.format(settings.MEDIA_URL, r.video_id,
                                                                                 r.detection_id) if r.detection_id else '{}{}/frames/{}.jpg'.format(
                                             settings.MEDIA_URL, r.video_id, r.frame.frame_index),
                                          'result_type': "Region" if r.detection_id else "Frame",
                                          'rank':r.rank,
                                          'frame_id': r.frame_id,
                                          'frame_index': r.frame.frame_index,
                                          'distance': r.distance,
                                          'video_id': r.video_id,
                                          'video_name': r.video.name}))
        for k, v in self.context.iteritems():
            if v:
                self.context[k].sort()
                self.context[k] = zip(*v)[1]

    def load_from_db(self,query,media_dir):
        self.query = query
        self.media_dir = media_dir

    def to_json(self):
        json_query = {
        }
        return json.dumps(json_query)

    def execute_sub_query(self,iq,visual_index):
        local_path = "{}/queries/{}_{}.png".format(self.media_dir, iq.algorithm, self.query.pk)
        with open(local_path, 'w') as fh:
            fh.write(str(self.query.image_data))
        vector = visual_index.apply(local_path)
        iq.vector = vector.tostring()
        iq.save()
        self.query.results_available = True
        self.query.save()
        return 0

    def perform_retrieval(self,iq,index_name,retrieval_task):
        retriever = retrieval_task.visual_retriever[index_name]
        exact = True
        results = []
        vector = np.fromstring(iq.vector)
        if iq.approximate:
            if retrieval_task.clusterer[index_name] is None:
                retrieval_task.load_clusterer(index_name)
            clusterer = retrieval_task.clusterer[index_name]
            if clusterer:
                results = query_approximate(retrieval_task, iq.count, retriever, clusterer)
                exact = False
        if exact:
            retrieval_task.refresh_index(index_name)
            results = retriever.nearest(vector=vector,n=iq.count)
        # TODO: optimize this using batching
        for r in results:
            qr = QueryResults()
            qr.query = self.query
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
        self.query.results_available = True
        self.query.save()
        return 0