import os
import logging
import time
import glob
import sys
from fabric.api import task, local
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/fab.log',
                    filemode='a')



@task
def copy_defaults():
    """
    Copy defaults from config volume
    :return:
    """
    if sys.platform == 'darwin':
        shutil.copy("../configs/custom_defaults/defaults_mac.py", 'dvaui/defaults.py')
    elif os.path.isfile("../configs/custom_defaults/defaults.py"):
        shutil.copy("../configs/custom_defaults/defaults.py",'dvaui/defaults.py')
    else:
        raise ValueError("defaults.py not found, if you have mounted a custom_config volume")


@task
def start_container(container_type):
    """
    Start container with queues launched as specified in environment
    """
    copy_defaults()
    init_fs()
    if container_type == 'worker':
        time.sleep(30)  # To avoid race condition where worker starts before migration is finished
        launch_workers_and_scheduler_from_environment(block_on_manager=True)
    elif container_type == 'server':
        launch_workers_and_scheduler_from_environment()
        launch_server_from_environment()
    else:
        raise ValueError, "invalid container_type = {}".format(container_type)

@task
def launch_workers_and_scheduler_from_environment(block_on_manager=False):
    """
    Launch workers and scheduler as specified in the environment variables.
    Only one scheduler should be launched per deployment.

    """
    import django, os
    if block_on_manager == '0':
        block_on_manager = False
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import TrainedModel, Retriever
    from django.conf import settings
    for k in os.environ:
        if k.startswith('LAUNCH_BY_NAME_'):
            qtype, model_name = k.split('_')[-2:]
            env_vars = ""
            if qtype == 'indexer':
                dm = TrainedModel.objects.get(name=model_name,model_type=TrainedModel.INDEXER)
                queue_name = 'q_indexer_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            elif qtype == 'retriever':
                dm = Retriever.objects.get(name=model_name)
                queue_name = 'q_retriever_{}'.format(dm.pk)
            elif qtype == 'detector':
                dm = TrainedModel.objects.get(name=model_name,model_type=TrainedModel.DETECTOR)
                queue_name = 'q_detector_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            elif qtype == 'analyzer':
                dm = TrainedModel.objects.get(name=model_name,model_type=TrainedModel.ANALYZER)
                queue_name = 'q_analyzer_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            else:
                raise ValueError, k
            command = '{}fab startq:{} &'.format(env_vars, queue_name)
            logging.info("'{}' for {}".format(command, k))
            local(command)
        elif k.startswith('LAUNCH_Q_') and k != 'LAUNCH_Q_{}'.format(settings.Q_MANAGER):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                local('fab startq:{},{} &'.format(queue_name, os.environ['LAUNCH_Q_qextract']))
            else:
                queue_name = k.split('_')[-1]
                local('fab startq:{} &'.format(queue_name))
    if os.environ.get("LAUNCH_SCHEDULER", False):
        # Should be launched only once per deployment
        local('fab start_scheduler &')
    if block_on_manager:  # the container process waits on the manager
        local('fab startq:{}'.format(settings.Q_MANAGER))
    else:
        local('fab startq:{} &'.format(settings.Q_MANAGER))


def launch_server_from_environment():
    """
    Launch django development server or NGINX server as specified in environment variable

    """
    if 'LAUNCH_SERVER' in os.environ:
        local('python manage.py runserver 0.0.0.0:8000')
    elif 'LAUNCH_SERVER_NGINX' in os.environ:
        local('chmod 0777 -R /tmp')
        try:
            local("mv ../configs/nginx.conf /etc/nginx/")
        except:
            print "warning assuming that the config was already moved"
            pass
        if 'ENABLE_BASICAUTH' in os.environ:
            try:
                local("mv ../configs/nginx-app_password.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        else:
            try:
                local("mv ../configs/nginx-app.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        try:
            local("mv ../configs/supervisor-app.conf /etc/supervisor/conf.d/")
        except:
            print "warning assuming that the config was already moved"
            pass
        local("python manage.py collectstatic --no-input")
        local("chmod 0777 -R dva/staticfiles/")
        # local("chmod 0777 -R dva/media/")
        local('supervisord -n')

@task
def init_fs():
    """
    Initialize filesystem by creating directories
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaui.models import ExternalServer
    from dvaapp.models import TrainedModel, Retriever, Video
    from dvaui.defaults import EXTERNAL_SERVERS
    from django.contrib.auth.models import User
    from django.utils import timezone
    from dvaui.defaults import DEFAULT_MODELS
    if not User.objects.filter(is_superuser=True).exists() and 'SUPERUSER' in os.environ:
        User.objects.create_superuser(username=os.environ['SUPERUSER'], password=os.environ['SUPERPASS'],
                                      email=os.environ['SUPEREMAIL'])
    for create_dirname in ['queries', 'exports', 'external', 'retrievers', 'ingest']:
        if not os.path.isdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname)):
            try:
                os.mkdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname))
            except:
                pass
    for e in EXTERNAL_SERVERS:
        ExternalServer.objects.get_or_create(name=e['name'],url=e['url'])
    for m in DEFAULT_MODELS:
        if m['model_type'] == "detector":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'],algorithm=m['algorithm'],mode=m['mode'],
                                                          files=m.get('files',[]), model_filename=m.get("filename", ""),
                                                          detector_type=m.get("detector_type", ""),
                                                          class_index_to_string=m.get("class_index_to_string", {}),
                                                          model_type=TrainedModel.DETECTOR,)
            if created:
                dm.download()
        if m['model_type'] == "indexer":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'], mode=m['mode'], files=m.get('files',[]),
                                                          shasum=m['shasum'],model_type=TrainedModel.INDEXER)
            if created:
                dr, dcreated = Retriever.objects.get_or_create(name=m['name'],
                                                               source_filters={'indexer_shasum': dm.shasum})
                if dcreated:
                    dr.last_built = timezone.now()
                    dr.save()
            if created:
                dm.download()
        if m['model_type'] == "analyzer":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'], files=m.get('files',[]), mode=m['mode'],
                                                          model_type=TrainedModel.ANALYZER)
            if created:
                dm.download()
    if 'TEST' in os.environ and Video.objects.count() == 0:
        test()



@task
def startq(queue_name, conc=3):
    """
    Start worker to handle a queue, Usage: fab startq:indexer
    Concurrency is set to 1 but you can edit code to change.
    :param conc:conccurency only for extractor

    """
    import django, os, shlex, subprocess
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    mute = '--without-gossip --without-mingle --without-heartbeat' if 'CELERY_MUTE' in os.environ else ''
    if queue_name == settings.Q_MANAGER:
        command = 'celery -A dva worker -l info {} -c 1 -Q qmanager -n manager.%h -f ../logs/qmanager.log'.format(mute)
    elif queue_name == settings.Q_EXTRACTOR:
        command = 'celery -A dva worker -l info {} -c {} -Q {} -n {}.%h -f ../logs/{}.log'.format(mute, max(int(conc), 2),
                                                                                               queue_name, queue_name,
                                                                                               queue_name)
        # TODO: worker fails due to
        # https://github.com/celery/celery/issues/3620
    else:
        command = 'celery -A dva worker -l info {} -P solo -c {} -Q {} -n {}.%h -f ../logs/{}.log'.format(mute, 1,
                                                                                                       queue_name,
                                                                                                       queue_name,
                                                                                                       queue_name)
    logging.info(command)
    c = subprocess.Popen(args=shlex.split(command))
    c.wait()


@task
def test():
    """
    Run tests by launching tasks

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file
    for fname in glob.glob('../tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name)
    for fname in glob.glob('../tests/ci/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)


@task
def start_scheduler():
    """
    Start celery-beat scheduler using django database as source for tasks.

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django_celery_beat.models import PeriodicTask,IntervalSchedule
    di,created = IntervalSchedule.objects.get_or_create(every=os.environ.get('REFRESH_MINUTES',3),period=IntervalSchedule.MINUTES)
    _ = PeriodicTask.objects.get_or_create(name="monitoring",task="monitor_system",interval=di,queue='qscheduler')
    local('fab startq:qscheduler &')
    local("celery -A dva beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler -f ../logs/beat.log")