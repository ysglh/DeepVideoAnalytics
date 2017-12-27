import base64, copy, os, subprocess, json, logging, time
from django.utils import timezone
from django.conf import settings
from dva.celery import app
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode")
from django.apps import apps
from models import Video,DVAPQL,TEvent,TrainedModel,Retriever,Worker
from celery.result import AsyncResult
import fs
import task_shared


SYNC_TASKS = {
    "perform_dataset_extraction":[{'operation':'perform_sync','arguments':{'dirname':'frames'}},],
    "perform_video_segmentation":[{'operation':'perform_sync','arguments':{'dirname':'segments'}},],
    "perform_video_decode":[{'operation': 'perform_sync', 'arguments': {'dirname': 'frames'}},],
    "perform_frame_download":[{'operation': 'perform_sync', 'arguments': {'dirname': 'frames'}},],
    'perform_detection':[],
    'perform_region_import':[],
    'perform_transformation':[{'operation': 'perform_sync', 'arguments': {'dirname': 'regions'}},],
    'perform_indexing':[{'operation': 'perform_sync', 'arguments': {'dirname': 'indexes'}},],
    'perform_import':[{'operation': 'perform_sync', 'arguments': {}},],
    'perform_detector_training':[],
    'perform_detector_import':[],
}

ANALYER_NAME_TO_PK = {}
INDEXER_NAME_TO_PK = {}
RETRIEVER_NAME_TO_PK = {}
DETECTOR_NAME_TO_PK = {}
CURRENT_QUEUES = set()
LAST_UPDATED = None


def refresh_queue_names():
    return {w.queue_name for w in Worker.objects.all().filter(alive=True)}


def get_queues():
    global CURRENT_QUEUES
    global LAST_UPDATED
    if LAST_UPDATED is None or (time.time() - LAST_UPDATED) > 120:
        CURRENT_QUEUES = refresh_queue_names()
        LAST_UPDATED = time.time()
    return CURRENT_QUEUES


def get_model_specific_queue_name(operation,args):
    if 'detector_pk' in args:
        queue_name = "q_detector_{}".format(args['detector_pk'])
    elif 'indexer_pk' in args:
        queue_name = "q_indexer_{}".format(args['indexer_pk'])
    elif 'retriever_pk' in args:
        queue_name = "q_retriever_{}".format(args['retriever_pk'])
    elif 'analyzer_pk' in args:
        queue_name = "q_analyzer_{}".format(args['analyzer_pk'])
    elif 'retriever' in args:
        if args['retriever'] not in RETRIEVER_NAME_TO_PK:
            RETRIEVER_NAME_TO_PK[args['retriever']] = Retriever.objects.get(name=args['retriever']).pk
        queue_name = 'q_retriever_{}'.format(RETRIEVER_NAME_TO_PK[args['retriever']])
    elif 'index' in args:
        if args['index'] not in INDEXER_NAME_TO_PK:
            INDEXER_NAME_TO_PK[args['index']] = TrainedModel.objects.get(name=args['index'],model_type=TrainedModel.INDEXER).pk
        queue_name = 'q_indexer_{}'.format(INDEXER_NAME_TO_PK[args['index']])
    elif 'analyzer' in args:
        if args['analyzer'] not in ANALYER_NAME_TO_PK:
            ANALYER_NAME_TO_PK[args['analyzer']] = TrainedModel.objects.get(name=args['analyzer'],model_type=TrainedModel.ANALYZER).pk
        queue_name = 'q_analyzer_{}'.format(ANALYER_NAME_TO_PK[args['analyzer']])
    elif 'detector' in args:
        if args['detector'] not in DETECTOR_NAME_TO_PK:
            DETECTOR_NAME_TO_PK[args['detector']] = TrainedModel.objects.get(name=args['detector'],model_type=TrainedModel.DETECTOR).pk
        queue_name = 'q_detector_{}'.format(DETECTOR_NAME_TO_PK[args['detector']])
    else:
        raise NotImplementedError,"{}, {}".format(operation,args)
    return queue_name


def get_model_pk_from_args(operation,args):
    if 'detector_pk' in args:
        return args['detector_pk']
    elif 'indexer_pk' in args:
        return args['indexer_pk']
    elif 'retriever_pk' in args:
        return args['retriever_pk']
    elif 'analyzer_pk' in args:
        return ['analyzer_pk']
    elif 'index' in args:
        if args['index'] not in INDEXER_NAME_TO_PK:
            INDEXER_NAME_TO_PK[args['index']] = TrainedModel.objects.get(name=args['index'],model_type=TrainedModel.INDEXER).pk
        return INDEXER_NAME_TO_PK[args['index']]
    elif 'analyzer' in args:
        if args['analyzer'] not in ANALYER_NAME_TO_PK:
            ANALYER_NAME_TO_PK[args['analyzer']] = TrainedModel.objects.get(name=args['analyzer'],model_type=TrainedModel.ANALYZER).pk
        return ANALYER_NAME_TO_PK[args['analyzer']]
    elif 'detector' in args:
        if args['detector'] not in DETECTOR_NAME_TO_PK:
            DETECTOR_NAME_TO_PK[args['detector']] = TrainedModel.objects.get(name=args['detector'],model_type=TrainedModel.DETECTOR).pk
        return DETECTOR_NAME_TO_PK[args['detector']]
    else:
        raise NotImplementedError,"{}, {}".format(operation,args)


def get_queue_name_and_operation(operation,args):
    global CURRENT_QUEUES
    if operation in settings.TASK_NAMES_TO_QUEUE:
        # Here we return directly since queue name is not per model
        return settings.TASK_NAMES_TO_QUEUE[operation], operation
    else:
        queue_name = get_model_specific_queue_name(operation,args)
        if queue_name not in CURRENT_QUEUES:
            CURRENT_QUEUES = refresh_queue_names()
        if queue_name not in CURRENT_QUEUES:
            if queue_name.startswith('q_retriever'):
                # Global retriever queue process all retrieval operations
                # If a worker processing the retriever queue does not exists send it to global
                if settings.GLOBAL_RETRIEVER_QUEUE_ENABLED:
                    return settings.GLOBAL_RETRIEVER, operation
                else:
                    return queue_name, operation
            else:
                # Check if global queue is enabled
                if settings.GLOBAL_MODEL_QUEUE_ENABLED:
                    # send it to a  global queue which loads model at every execution
                    return settings.GLOBAL_MODEL, operation
        return queue_name, operation


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
        q, op = get_queue_name_and_operation(k['operation'], args)
        next_task = TEvent.objects.create(video=v, operation=op, arguments=args, parent=dt, parent_process=p, queue=q)
        tids.append(app.send_task(k['operation'], args=[next_task.pk, ], queue=q).id)
    return tids


def process_next(task_id,inject_filters=None,custom_next_tasks=None,sync=True,launch_next=True):
    if custom_next_tasks is None:
        custom_next_tasks = []
    dt = TEvent.objects.get(pk=task_id)
    launched = []
    logging.info("next tasks for {}".format(dt.operation))
    next_tasks = dt.arguments.get('map',[]) if dt.arguments and launch_next else []
    if sync and settings.MEDIA_BUCKET:
        for k in SYNC_TASKS.get(dt.operation,[]):
            if settings.DISABLE_NFS:
                dirname = k['arguments'].get('dirname',None)
                task_shared.upload(dirname,task_id,dt.video_id)
            else:
                launched += launch_tasks(k,dt,inject_filters,None,'sync')
    for k in next_tasks+custom_next_tasks:
        map_filters = get_map_filters(k,dt.video)
        launched += launch_tasks(k, dt, inject_filters,map_filters,'map')
    return launched


def mark_as_completed(start):
    start.completed = True
    if start.start_ts:
        start.duration = (timezone.now() - start.start_ts).total_seconds()
    start.save()


class DVAPQLProcess(object):

    def __init__(self,process=None,media_dir=None):
        self.process = process
        self.media_dir = media_dir
        self.task_results = {}
        self.created_objects = []

    def store(self):
        query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.process.pk)
        with open(query_path, 'w') as fh:
            fh.write(self.process.image_data)
        if settings.DISABLE_NFS:
            query_key = "/queries/{}.png".format(self.process.pk)
            fs.upload_file_to_remote(query_key)
            os.remove(query_path)

    def create_from_json(self, j, user=None):
        if self.process is None:
            self.process = DVAPQL()
        if not (user is None):
            self.process.user = user
        if j['process_type'] == DVAPQL.QUERY:
            if j['image_data_b64'].strip():
                image_data = base64.decodestring(j['image_data_b64'])
                j['image_data_b64'] = None
                self.process.image_data = image_data
            self.process.process_type = DVAPQL.QUERY
            self.process.script = j
            self.process.save()
            self.store()
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
            for c in self.process.script.get('create',[]):
                m = apps.get_model(app_label='dvaapp',model_name=c['MODEL'])
                for k,v in c['spec'].iteritems():
                    if v == '__timezone.now__':
                        c['spec'][k] = timezone.now()
                instance = m.objects.create(**c['spec'])
                self.created_objects.append(instance)
                for t in copy.deepcopy(c.get('tasks',[])):
                    self.launch_task(t,instance.pk)
            for t in self.process.script.get('tasks',[]):
                self.launch_task(t)
        elif self.process.script['process_type'] == DVAPQL.QUERY:
            for t in self.process.script['tasks']:
                operation = t['operation']
                arguments = t.get('arguments',{})
                queue_name, operation = get_queue_name_and_operation(operation,arguments)
                next_task = TEvent.objects.create(parent_process=self.process, operation=operation,arguments=arguments,queue=queue_name)
                self.task_results[next_task.pk] = app.send_task(name=operation,args=[next_task.pk, ],queue=queue_name,priority=5)
        else:
            raise NotImplementedError

    def wait(self,timeout=60):
        for _, result in self.task_results.iteritems():
            try:
                next_task_ids = result.get(timeout=timeout)
                if next_task_ids:
                    for next_task_id in next_task_ids:
                        next_result = AsyncResult(id=next_task_id)
                        _ = next_result.get(timeout=timeout)
            except Exception, e:
                raise ValueError(e)

    def launch_task(self,t,created_pk=None):
        if created_pk:
            if t.get('video_id','') == '__pk__':
                t['video_id'] = created_pk
            for k, v in t.get('arguments',{}).iteritems():
                if v == '__pk__':
                    t['arguments'][k] = created_pk
        if 'video_id' in t:
            v = Video.objects.get(pk=t['video_id'])
            map_filters = get_map_filters(t, v)
        else:
            map_filters = [{}]
        for f in map_filters:
            args = copy.deepcopy(t.get('arguments', {}))  # make copy so that spec isnt mutated.
            if f:
                if 'filters' not in args:
                    args['filters'] = f
                else:
                    args['filters'].update(f)
            dt = TEvent()
            dt.parent_process = self.process
            if 'video_id' in t:
                dt.video_id = t['video_id']
            dt.arguments = args
            dt.queue, op = get_queue_name_and_operation(t['operation'], t.get('arguments', {}))
            dt.operation = op
            dt.save()
            self.task_results[dt.pk] = app.send_task(name=dt.operation, args=[dt.pk, ], queue=dt.queue)

    def to_json(self):
        json_query = {}
        return json.dumps(json_query)


def defer(start):
    """
    Check if a worker that processes model specific queue has become available.
    :param start:
    :return:
    """
    model_specific_queue_name = get_model_specific_queue_name(start.operation, start.arguments)
    if model_specific_queue_name in get_queues():
        start.started = False
        start.queue_name = model_specific_queue_name
        start.start_ts = None
        start.worker = None
        start.save()
        app.send_task(start.task_name, args=[start.pk, ], queue=model_specific_queue_name)
        return True
    return False


def run_task_in_new_process(start):
    """
    Run in a new process,
    TODO: Make sure that the new process has correct environment variable mode.
    E.g. TF or PyTorch, otherwise it won't be able to import necessary library
    :param start:
    :return:
    """
    model_specific_queue_name = get_model_specific_queue_name(start.operation, start.arguments)
    trained_model = TrainedModel.objects.get(pk=get_model_pk_from_args(start.operation,start.arguments))
    start.started = False
    start.queue_name = model_specific_queue_name
    start.start_ts = None
    new_envs = os.environ.copy()
    for k in {'PYTORCH_MODE','CAFFE_MODE','MXNET_MODE'}:
        if k in new_envs:
            del new_envs[k]
    if trained_model.mode == TrainedModel.PYTORCH:
        new_envs['PYTORCH_MODE'] = '1'
    elif trained_model.mode == TrainedModel.CAFFE:
        new_envs['CAFFE_MODE'] = '1'
    elif trained_model.mode == TrainedModel.MXNET:
        new_envs['MXNET_MODE'] = '1'
    s = subprocess.Popen(['python', 'scripts/run_task.py', start.operation, str(start.pk)],env=new_envs)
    s.wait()
    if s.returncode != 0:
        raise ValueError("run_task.py failed")
    return True
