import base64, copy
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

from ..models import Video,DVAPQL,IndexerQuery,QueryResults,TEvent,Region
from collections import defaultdict
from celery.result import AsyncResult


SYNC_TASKS = {
    "perform_dataset_extraction":[{'operation':'perform_sync','arguments':{'dirname':'frames'}},],
    "perform_video_segmentation":[{'operation':'perform_sync','arguments':{'dirname':'segments'}},],
    "perform_video_decode":[{'operation': 'perform_sync', 'arguments': {'dirname': 'frames'}},],
    'perform_detection':[],
    'perform_transformation':[{'operation': 'perform_sync', 'arguments': {'dirname': 'regions'}},],
    'perform_indexing':[{'operation': 'perform_sync', 'arguments': {'dirname': 'indexes'}},],
    'perform_import':[{'operation': 'perform_sync', 'arguments': {}},],
    'perform_detector_training':[],
    'perform_detector_import':[],
}


def get_queue_name(operation,args):
    if operation in settings.TASK_NAMES_TO_QUEUE:
        return settings.TASK_NAMES_TO_QUEUE[operation]
    elif 'index' in args and operation == 'perform_retrieval':
        return settings.VISUAL_INDEXES[args['index']]['retriever_queue']
    elif 'index' in args:
        return settings.VISUAL_INDEXES[args['index']]['indexer_queue']
    elif 'analyzer' in args:
        return settings.ANALYZERS[args['analyzer']]['queue']
    elif 'detector' in args:
        if args['detector'] == 'custom':
            return "qcustomdetector_{}".format(args['detector_pk']) # route it according to the custom detector
        else:
            return settings.DETECTORS[args['detector']]['queue']
    else:
        raise NotImplementedError,"{}, {}".format(operation,args)


def perform_substitution(args,parent_task,inject_filters,map_filters):
    """
    Its important to do a deep copy of args before executing any mutations.
    :param args:
    :param parent_task:
    :return:
    """
    args = copy.deepcopy(args) # IMPORTANT otherwise the first task to execute on the worker will fill the filters
    inject_filters = copy.deepcopy(inject_filters) # IMPORTANT otherwise the first task to execute on the worker will fill the filters
    map_filters = copy.deepcopy(map_filters) # IMPORTANT otherwise the first task to execute on the worker will fill the filters
    filters = args.get('filters',{})
    parent_args = parent_task.arguments
    if filters == '__parent__':
        parent_filters = parent_args.get('filters',{})
        logging.info('using filters from parent arguments: {}'.format(parent_args))
        args['filters'] = parent_filters
    elif filters:
        for k,v in args.get('filters',{}).items():
            if v == '__parent_event__':
                args['filters'][k] = parent_task.pk
            elif v == '__grand_parent_event__':
                args['filters'][k] = parent_task.parent.pk
    if inject_filters:
        if 'filters' not in args:
            args['filters'] = inject_filters
        else:
            args['filters'].update(inject_filters)
    if map_filters:
        if 'filters' not in args:
            args['filters'] = map_filters
        else:
            args['filters'].update(map_filters)
    return args


def get_map_filters(k, v):
    """
    TO DO add vstart=0,vstop=None
    """
    vstart = 0
    map_filters = []
    if 'segments_batch_size' in k['arguments']:
        step = k['arguments']["segments_batch_size"]
        vstop = v.segments
        for gte, lt in [(start, start + step) for start in range(vstart, vstop, step)]:
            if lt < v.segments:
                map_filters.append({'segment_index__gte': gte, 'segment_index__lt': lt})
            else:  # ensures off by one error does not happens [gte->
                map_filters.append({'segment_index__gte': gte})
    elif 'frames_batch_size' in k['arguments']:
        step = k['arguments']["frames_batch_size"]
        vstop = v.frames
        for gte, lt in [(start, start + step) for start in range(vstart, vstop, step)]:
            if lt < v.frames:  # to avoid off by one error
                map_filters.append({'frame_index__gte': gte, 'frame_index__lt': lt})
            else:
                map_filters.append({'frame_index__gte': gte})
    else:
        map_filters.append({})  # append an empty filter
    # logging.info("Running with map filters {}".format(map_filters))
    return map_filters


def launch_tasks(k, dt, inject_filters, map_filters = None, launch_type = ""):
    v = dt.video
    op = k['operation']
    p = dt.parent_process
    if map_filters is None:
        map_filters = [{},]
    tids = []
    for f in map_filters:
        args = perform_substitution(k['arguments'], dt, inject_filters, f)
        logging.info("launching {} -> {} with args {} as specified in {}".format(dt.operation, op, args, launch_type))
        q = get_queue_name(k['operation'], args)
        next_task = TEvent.objects.create(video=v, operation=op, arguments=args, parent=dt, parent_process=p, queue=q)
        tids.append(app.send_task(k['operation'], args=[next_task.pk, ], queue=q).id)
    return tids


def process_next(task_id,inject_filters=None,custom_next_tasks=None,sync=True,launch_next=True):
    if custom_next_tasks is None:
        custom_next_tasks = []
    dt = TEvent.objects.get(pk=task_id)
    launched = []
    logging.info("next tasks for {}".format(dt.operation))
    next_tasks = dt.arguments.get('next_tasks',[]) if dt.arguments and launch_next else []
    if sync and settings.MEDIA_BUCKET:
        for k in SYNC_TASKS.get(dt.operation,[]):
            launched += launch_tasks(k,dt,inject_filters,None,'sync')
    for k in next_tasks+custom_next_tasks:
        map_filters = get_map_filters(k,dt.video)
        launched += launch_tasks(k, dt, inject_filters,map_filters,'next_tasks')
    return launched


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
                if 'video_id' in t:
                    v = Video.objects.get(pk=t['video_id'])
                    map_filters = get_map_filters(t,v)
                else:
                    map_filters = [{}]
                for f in map_filters:
                    args = copy.deepcopy(t.get('arguments',{})) # make copy so that spec isnt mutated.
                    if f:
                        if 'filters' not in args:
                            args['filters'] = f
                        else:
                            args['filters'].update(f)
                    dt = TEvent()
                    dt.parent_process = self.process
                    if 'video_id' in t:
                        dt.video_id = t['video_id']
                    dt.operation = t['operation']
                    dt.arguments = args
                    dt.queue = get_queue_name(t['operation'],t.get('arguments',{}))
                    dt.save()
                    app.send_task(name=dt.operation,args=[dt.pk, ],queue=dt.queue)
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
                queue_name = get_queue_name(operation, jargs)
                next_task = TEvent.objects.create(parent_process=self.process, operation=operation, arguments=jargs,queue=queue_name)
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



