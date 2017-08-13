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

def get_queue_name(operation,args):
    if operation in settings.TASK_NAMES_TO_QUEUE:
        return settings.TASK_NAMES_TO_QUEUE[operation]
    elif 'index' in args and operation == 'perform_retrieval':
        return settings.VISUAL_INDEXES[args['index']]['retriever_queue']
    elif 'index' in args:
        return settings.VISUAL_INDEXES[args['index']]['indexer_queue']
    elif 'detector' in args:
        return settings.DETECTORS[args['detector']]['queue']
    else:
        raise NotImplementedError,"{}, {}".format(operation,args)


class DVAPQLProcess(object):

    def __init__(self,process=None,media_dir=None):
        self.process = process
        self.media_dir = media_dir
        self.task_results = {}
        self.context = {}
        self.visual_indexes = settings.VISUAL_INDEXES

    def store(self):
        if settings.HEROKU_DEPLOY:
            query_key = "queries/{}.png".format(self.process.pk)
            s3 = boto3.resource('s3')
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_key, Body=self.process.image_data)
        else:
            query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.process.pk)
            with open(query_path, 'w') as fh:
                fh.write(self.process.image_data)

    def create_from_request(self, request):
        """
        Create JSON object representing the query from request recieved from Dashboard.
        :param request:
        :return:
        """
        query_json = {'process_type':DVAPQL.QUERY}
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
        return self.process

    def create_from_json(self, j, user=None):
        if self.process is None:
            self.process = DVAPQL()
        if not (user is None):
            self.process.user = user
        if j['process_type'] == DVAPQL.QUERY:
            if j['image_data_b64'].strip():
                image_data = base64.decodestring(j['image_data_b64'])
                self.process.image_data = image_data
            self.process.script = j
            self.process.save()
            self.store()
            for k in j['indexer_queries']:
                iq = IndexerQuery()
                iq.parent_query = self.process
                iq.algorithm = k['algorithm']
                iq.count = k['count']
                iq.excluded_index_entries_pk = k['excluded_index_entries_pk'] if 'excluded_index_entries_pk' in k else []
                iq.approximate = k['approximate']
                iq.save()
        elif j['process_type'] == DVAPQL.PROCESS:
            self.process.process_type = DVAPQL.PROCESS
            self.process.script = j
            self.process.save()
        elif j['process_type'] == DVAPQL.INGEST:
            raise NotImplementedError
        else:
            raise ValueError
        return self.process

    def validate(self):
        pass

    def launch(self):
        if self.process.script['process_type'] == DVAPQL.PROCESS:
            for t in self.process.script['tasks']:
                dt = TEvent()
                dt.parent_process = self.process
                if 'video_id' in t:
                    dt.video_id = t['video_id']
                dt.operation = t['operation']
                dt.arguments = t.get('arguments',{})
                dt.save()
                app.send_task(name=dt.operation,
                              args=[dt.pk, ],
                              queue=get_queue_name(dt.operation,dt.arguments))
        elif self.process.script['process_type'] == DVAPQL.QUERY:
            for iq in IndexerQuery.objects.filter(parent_query=self.process):
                operation = 'perform_indexing'
                jargs = {
                    'iq_id':iq.pk,
                    'index':iq.algorithm,
                    'target':'query',
                    'next_tasks':[
                        { 'operation': 'perform_retrieval',
                          'arguments': {'iq_id': iq.pk,'index':iq.algorithm}
                         }
                    ]
                }
                next_task = TEvent.objects.create(parent_process=self.process, operation=operation, arguments=jargs)
                queue_name = get_queue_name(next_task.operation, next_task.arguments)
                self.task_results[iq.algorithm] = app.send_task(operation, args=[next_task.pk, ], queue=queue_name, priority=5)
                self.context[iq.algorithm] = []
        else:
            raise NotImplementedError

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
        for r in QueryResults.objects.all().filter(query=self.process):
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



