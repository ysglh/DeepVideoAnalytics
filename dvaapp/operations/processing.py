import base64
import json,logging
import boto3
from django.conf import settings
from dva.celery import app
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from ..models import Video,DVAPQL,IndexerQuery,QueryResults,TEvent
from collections import defaultdict
from celery.result import AsyncResult


class DVAPQLProcess(object):

    def __init__(self,query=None,media_dir=None):
        self.query = query
        self.query_json = {}
        self.media_dir = media_dir
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
        """
        Create JSON object representing the query from request recieved from Dashboard.
        :param request:
        :return:
        """
        query_json = {}
        count = request.POST.get('count')
        excluded = json.loads(request.POST.get('excluded_index_entries'))
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
        approximate = True if request.POST.get('approximate') == 'true' else False
        query_json['image_data_b64'] = request.POST.get('image_url')[22:]
        query_json['indexer_queries'] = []
        for k in selected_indexers:
            query_json['indexer_queries'].append({
                'algorithm':k,
                'count':count,
                'excluded_index_entries_pk': [int(epk) for epk in excluded] if excluded else [],
                'approximate':approximate
            })
        user = request.user if request.user.is_authenticated else None
        self.create_from_json(query_json,user)
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
        if self.query is None:
            self.query = DVAPQL()
        if not (user is None):
            self.query.user = user
        if j['image_data_b64'].strip():
            image_data = base64.decodestring(j['image_data_b64'])
            self.query.image_data = image_data
        self.query.query_json = j
        self.query.save()
        self.store_and_create_video_object()
        for k in j['indexer_queries']:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k['algorithm']
            iq.count = k['count']
            iq.excluded_index_entries_pk = k['excluded_index_entries_pk'] if 'excluded_index_entries_pk' in k else []
            iq.approximate = k['approximate']
            iq.save()
        return self.query

    def validate(self):
        pass

    def launch(self):
        for iq in IndexerQuery.objects.filter(parent_query=self.query):
            task_name = 'perform_indexing'
            queue_name = self.visual_indexes[iq.algorithm]['indexer_queue']
            jargs = {
                'iq_id':iq.pk,
                'index':iq.algorithm,
                'target':'query',
                'next_tasks':[
                    { 'task_name': 'perform_retrieval',
                      'arguments': {'iq_id': iq.pk,'index':iq.algorithm}
                     }
                ]
            }
            next_task = TEvent.objects.create(video=self.dv, operation=task_name, arguments_json=jargs)
            self.task_results[iq.algorithm] = app.send_task(task_name, args=[next_task.pk, ], queue=queue_name, priority=5)
            self.context[iq.algorithm] = []

    def wait(self,timeout=60):
        for visual_index_name, result in self.task_results.iteritems():
            try:
                next_task_ids = result.get(timeout=timeout)
                if next_task_ids:
                    for next_task_id in next_task_ids:
                        next_result = AsyncResult(id=next_task_id)
                        _ = next_result.get(timeout=timeout)
            except Exception, e:
                raise ValueError(e)

    def collect(self):
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

    def to_json(self):
        json_query = {}
        return json.dumps(json_query)



