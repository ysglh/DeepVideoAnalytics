import subprocess,os,requests, signal, time
from django.conf import settings
from . import processing
from .models import TrainedModel
from dva.celery import app

GLOBAL_FLASK_SERVER_PROCESS = None
LAST_GLOBAL_QUEUE_NAME = None


def defer(start):
    """
    Check if a worker that processes model specific queue has become available.
    :param start:
    :return:
    """
    model_specific_queue_name = processing.get_model_specific_queue_name(start.operation, start.arguments)
    if model_specific_queue_name in processing.get_queues():
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
    trained_model = TrainedModel.objects.get(pk=processing.get_model_pk_from_args(start.operation,start.arguments))
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


def run_task_in_model_specific_flask_server(start):
    """
    Run in a new flask server,
    :param start:
    :return:
    """
    global GLOBAL_FLASK_SERVER_PROCESS
    global LAST_GLOBAL_QUEUE_NAME
    model_specific_queue_name = processing.get_model_specific_queue_name(start.operation, start.arguments)
    trained_model = TrainedModel.objects.get(pk=processing.get_model_pk_from_args(start.operation,start.arguments))
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
    if GLOBAL_FLASK_SERVER_PROCESS is None or LAST_GLOBAL_QUEUE_NAME != model_specific_queue_name:
        if GLOBAL_FLASK_SERVER_PROCESS:
            GLOBAL_FLASK_SERVER_PROCESS.terminate()
            os.remove('flask.pid')
        elif os.path.isfile('flask.pid'):
            try:
                os.kill(int(file('flask.pid').read()),signal.SIGTERM)
            except:
                pass
        GLOBAL_FLASK_SERVER_PROCESS = subprocess.Popen(['python', 'scripts/run_flask.py',
                                                        start.operation, str(start.pk)],env=new_envs)
        LAST_GLOBAL_QUEUE_NAME = model_specific_queue_name
        max_attempts = 15
        while max_attempts:
            try:
                r = requests.get('http://localhost:{port}/'.format(port=settings.GLOBAL_MODEL_FLASK_SERVER_PORT))
                if r.ok:
                    break
            except:
                max_attempts -= 1
                time.sleep(4)
    r = requests.get('http://localhost:{port}/{pk}/'.format(port=settings.GLOBAL_MODEL_FLASK_SERVER_PORT,pk=start.pk))
    if not r.ok:
        raise ValueError("Coud not process")
    return True
