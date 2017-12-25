import os
import logging
import time
import glob
import subprocess
import calendar
import sys
from fabric.api import task, local, lcd
import json
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/fab.log',
                    filemode='a')


@task
def local_static():
    """
    Collect static
    """
    local('python manage.py collectstatic')


@task
def migrate():
    """
    Make migrations and migrate database
    """
    local('python manage.py makemigrations')
    local('python manage.py migrate')


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
    if container_type == 'worker':
        time.sleep(30)  # To avoid race condition where worker starts before migration is finished
        init_fs()
        init_models()
        launch_workers_and_scheduler_from_environment(block_on_manager=True)
    elif container_type == 'server':
        init_fs()
        init_server()
        init_models()
        launch_workers_and_scheduler_from_environment()
        launch_server_from_environment()
    else:
        raise ValueError, "invalid container_type = {}".format(container_type)


@task
def clean():
    """
    Reset database, migrate, clear media folder, and (only on my dev machine) kill workers/clear all queues.
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    if settings.DEV_ENV:
        for qname in set(settings.TASK_NAMES_TO_QUEUE.values()):
            try:
                local('rabbitmqadmin purge queue name={}'.format(qname))
            except:
                logging.warning("coudnt clear queue {}".format(qname))
    # TODO: wait for Celery bug fix https://github.com/celery/celery/issues/3620
    # local('celery amqp exchange.delete broadcast_tasks')
    migrate()
    local('python manage.py flush --no-input')
    migrate()
    shutil.copy("../configs/custom_defaults/defaults_mac.py", 'dvaui/defaults.py')
    for dirname in os.listdir(settings.MEDIA_ROOT):
        if dirname != 'gitkeep':
            local("rm -rf {}/{}".format(settings.MEDIA_ROOT,dirname))
    if settings.DEV_ENV:
        local("rm ../logs/*.log")
        try:
            local("ps auxww | grep 'celery -A dva' | awk '{print $2}' | xargs kill -9")
        except:
            pass
        os.environ['SUPERUSER'] = 'akshay'
        os.environ['SUPERPASS'] = 'super'
        os.environ['SUPEREMAIL'] = 'test@deepvideoanalytics.com'
    init_fs()
    init_server()
    init_models()


@task
def restart_queues():
    """
    Kill all workers and launch them again

    """
    kill()
    launch()


@task
def kill():
    try:
        local("ps auxww | grep 'celery -A dva * ' | awk '{print $2}' | xargs kill -9")
    except:
        pass


@task
def quick():
    """
    Clear and launch for testing on dev machine

    """
    clean()
    test()
    launch()


@task
def launch():
    """
    Launch workers on dev machine by adding environment variables

    """
    envars = ['LAUNCH_BY_NAME_indexer_inception', 'LAUNCH_BY_NAME_indexer_facenet',
              'LAUNCH_BY_NAME_retriever_inception', 'LAUNCH_BY_NAME_retriever_facenet',
              'LAUNCH_BY_NAME_detector_coco', 'LAUNCH_BY_NAME_detector_face', 'LAUNCH_BY_NAME_analyzer_tagger',
              'LAUNCH_Q_qclusterer', 'LAUNCH_Q_qextract','LAUNCH_SCHEDULER']
    for k in envars:
        os.environ[k] = "1"
    # if sys.platform == 'darwin':
    #     os.environ['MEDIA_BUCKET'] = 'aub3dvatest'
    #     os.environ['DISABLE_NFS'] = '1'

    launch_workers_and_scheduler_from_environment(False)


@task
def launch_workers_and_scheduler_from_environment(block_on_manager=False):
    """
    Launch workers and scheduler as specified in the environment variables.
    Only one scheduler should be launched per deployment.

    """
    import django, os
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
def init_server():
    """
    Initialize server database by adding default DVAPQL templates
 
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video
    if 'TEST' in os.environ and Video.objects.count() == 0:
        test()



@task
def init_models():
    """
    Initialize default models in database specified in models.json,
    and download models to filesystem. Models are downloaded even if the database
    entries exist , but files doe not  since the worker might not be running in shared filesystem model.
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.utils import timezone
    from dvaapp.models import TrainedModel, Retriever
    from dvaui.defaults import DEFAULT_MODELS
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
    from dvaui.defaults import EXTERNAL_SERVERS
    from django.contrib.auth.models import User
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
def backup():
    """
    Take a backup, backups are store as a single zip file in backups/ folder

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    try:
        os.mkdir('backups')
    except:
        pass
    media_dir = settings.MEDIA_ROOT
    db = settings.DATABASES.values()[0]
    pg = '/Users/aub3/PostgreSQL/pg96/bin/pg_dump' if sys.platform == 'darwin' else 'pg_dump'
    with open('{}/postgres.dump'.format(media_dir), 'w') as dumpfile:
        dump = subprocess.Popen([pg, '--clean', '--dbname',
                                 'postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'],
                                                                        db['NAME'])], cwd=media_dir, stdout=dumpfile)
        dump.communicate()
    print dump.returncode
    current_path = os.path.abspath(os.path.dirname(__file__))
    command = ['zip', '-r', '{}/backups/backup_{}.zip'.format(current_path, calendar.timegm(time.gmtime())), '.']
    print ' '.join(command)
    zipper = subprocess.Popen(command, cwd=media_dir)
    zipper.communicate()
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode


@task
def restore(path):
    """
    Restore a backup using path provided. Note that arugment are provided in following format. fab restore:backups/backup_1.zip
    :param path:

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    media_dir = settings.MEDIA_ROOT
    current_path = os.path.abspath(os.path.dirname(__file__))
    command = ['unzip', '-o', '{}'.format(os.path.join(current_path, path))]
    print ' '.join(command)
    zipper = subprocess.Popen(command, cwd=media_dir)
    zipper.communicate()
    db = settings.DATABASES.values()[0]
    pg = '/Users/aub3/PostgreSQL/pg96/bin/psql' if sys.platform == 'darwin' else 'psql'
    with open('{}/postgres.dump'.format(media_dir)) as dumpfile:
        dump = subprocess.Popen(
            [pg, '--dbname',
             'postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'], db['NAME'])],
            cwd=media_dir, stdin=dumpfile)
        dump.communicate()
    print dump.returncode
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode


@task
def qt():
    """
    Add short videos/datasets and launch default tasks for quick testing
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file
    for fname in glob.glob('../tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/mp4")
        _ = handle_uploaded_file(f, name)
        break
    for fname in glob.glob('../tests/ci/example*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        _ = handle_uploaded_file(f, name)
        break



@task
def submit(path):
    """
    Submit a DVAPQL process to run
    :param path:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.processing import DVAPQLProcess
    with open(path) as f:
        j = json.load(f)
    p = DVAPQLProcess()
    p.create_from_json(j)
    p.launch()
    print "launched Process with id {} ".format(p.process.pk)


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


@task
def store_token_for_testing():
    """
    Generate & store authentication token for superuser (akshay) to test REST API.
    """
    import django, uuid
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.contrib.auth.models import User
    from rest_framework.authtoken.models import Token
    u = User.objects.all().first()
    if u is None:
        u = User.objects.create_user("test_token_user",email="test@test.com",password=str(uuid.uuid1()))
    token, _ = Token.objects.get_or_create(user=u)
    with open('creds.json', 'w') as creds:
        creds.write(json.dumps({'token': token.key}))
