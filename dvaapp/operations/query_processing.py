import logging, json, base64
import boto3
from django.conf import settings
import celery
from dva.celery import app

from dvalib import indexer,clustering
from ..models import IndexEntries,Clusters,Video,Query,IndexerQuery,QueryResults,Region,ClusterCodes


class IndexerTask(celery.Task):
    _visual_indexer = None
    _clusterer = None

    @property
    def visual_indexer(self):
        if IndexerTask._visual_indexer is None:
            IndexerTask._visual_indexer = {'inception': indexer.InceptionIndexer(),
                                           'facenet': indexer.FacenetIndexer(),
                                           'alexnet': indexer.AlexnetIndexer()}
        return IndexerTask._visual_indexer

    @property
    def clusterer(self):
        if IndexerTask._clusterer is None:
            IndexerTask._clusterer = {'inception': None, 'facenet': None, 'alexnet': None}
        return IndexerTask._clusterer

    def refresh_index(self, index_name):
        index_entries = IndexEntries.objects.all()
        visual_index = self.visual_indexer[index_name]
        for index_entry in index_entries:
            if index_entry.pk not in visual_index.loaded_entries and index_entry.algorithm == index_name:
                fname = "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id,
                                                  index_entry.features_file_name)
                vectors = indexer.np.load(fname)
                vector_entries = json.load(file("{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id,
                                                                          index_entry.entries_file_name)))
                logging.info("Starting {} in {}".format(index_entry.video_id, visual_index.name))
                start_index = visual_index.findex
                try:
                    visual_index.load_index(vectors, vector_entries)
                except:
                    logging.info("ERROR Failed to load {} ".format(index_entry.video_id))
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
    results = {}
    results[visual_index.name] = []
    coarse, fine, results_indexes = clusterer.apply(vector, n)
    for i, k in enumerate(results_indexes[0]):
        e = ClusterCodes.objects.get(searcher_index=k.id, clusters=clusterer.dc)
        if e.detection_id:
            results[visual_index.name].append({
                'rank': i + 1,
                'dist': i,
                'detection_primary_key': e.detection_id,
                'frame_index': e.frame.frame_index,
                'frame_primary_key': e.frame_id,
                'video_primary_key': e.video_id,
                'type': 'detection',
            })
        else:
            results[visual_index.name].append({
                'rank': i + 1,
                'dist': i,
                'detection_primary_key': e.detection_id,
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
        self.visual_indexes = settings.VISUAL_INDEXES

    def store_and_create_video_object(self):
        dv = Video()
        dv.name = 'query_{}'.format(self.query.pk)
        dv.dataset = True
        dv.query = True
        dv.parent_query = self.query
        dv.save()
        if settings.HEROKU_DEPLOY:
            query_key = "queries/{}.png".format(self.query.pk)
            query_frame_key = "{}/frames/0.png".format(dv.pk)
            s3 = boto3.resource('s3')
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_key, Body=self.query.image_data)
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_frame_key, Body=self.query.image_data)
        else:
            query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.query.pk)
            query_frame_path = "{}/{}/frames/0.png".format(settings.MEDIA_ROOT, dv.pk)
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
            task_name = 'execute_index_subquery'
            queue_name = self.visual_indexes[iq.algorithm]['retriever_queue']
            self.task_results[iq.algorithm] = app.send_task(task_name, args=[iq.pk, ], queue=queue_name)
            self.context[iq.algorithm] = []

    def wait(self,timeout=120):
        for visual_index_name, result in self.task_results.iteritems():
            try:
                logging.info("Waiting for {}".format(visual_index_name))
                _ = result.get(timeout=timeout)
            except Exception, e:
                raise ValueError(e)

    def collect_results(self):
        for r in QueryResults.objects.all().filter(query=self.query):
            self.context[r.algorithm].append((r.rank,
                                         {'url': '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL, r.video_id,
                                                                                 r.detection_id) if r.detection_id else '{}{}/frames/{}.jpg'.format(
                                             settings.MEDIA_URL, r.video_id, r.frame.frame_index),
                                          'result_type': "Region" if r.detection_id else "Frame",
                                          'rank':r.rank,
                                          'frame_id': r.frame_id,
                                          'frame_index': r.frame.frame_index,
                                          'video_id': r.video_id,
                                          'video_name': r.video.name}))
        for k, v in self.context.iteritems():
            if v:
                self.context[k].sort()
                self.context[k] = zip(*v)[1]

    def load_from_db(self,query,media_dir):
        self.query = query
        self.media_dir = media_dir

    def execute_sub_query(self,iq,index_name,query_task):
        visual_index = query_task.visual_indexer[index_name]
        exact = True
        local_path = "{}/queries/{}_{}.png".format(self.media_dir, iq.algorithm, self.query.pk)
        with open(local_path, 'w') as fh:
            fh.write(str(self.query.image_data))
        results = []
        if iq.approximate:
            if query_task.clusterer[index_name] is None:
                query_task.load_clusterer(index_name)
            clusterer = query_task.clusterer[index_name]
            if clusterer:
                results = query_approximate(local_path, iq.count, visual_index, clusterer)
                exact = False
        if exact:
            query_task.refresh_index(index_name)
            results = visual_index.nearest(image_path=local_path)
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