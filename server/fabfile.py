import logging, time, sys, os, shutil, shlex, subprocess
from fabric.api import task, local


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
def start_container(container_type='server'):
    """
    Start container with queues launched as specified in environment
    """
    copy_defaults()
    init_fs()
    block_on_manager = False
    if container_type == 'worker':
        block_on_manager = True
        time.sleep(30)  # To avoid race condition where worker starts before migration is finished
    launch_workers_and_scheduler_from_environment(block_on_manager=block_on_manager)


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
            command = '{} ./startq.py {} &'.format(env_vars, queue_name)
            logging.info("'{}' for {}".format(command, k))
            local(command)
        elif k.startswith('LAUNCH_Q_') and k != 'LAUNCH_Q_{}'.format(settings.Q_MANAGER):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                local('./startq.py {} {} &'.format(queue_name, os.environ['LAUNCH_Q_qextract']))
            elif k.startswith('LAUNCH_Q_qglobal'):
                queue_name = k.strip('LAUNCH_Q_')
                local('./startq.py {} &'.format(queue_name))
            else:
                queue_name = k.split('_')[-1]
                local('./startq.py {} &'.format(queue_name))
    if os.environ.get("LAUNCH_SCHEDULER", False):
        # Should be launched only once per deployment
        local('fab start_scheduler &')
    if block_on_manager:  # the container process waits on the manager
        local('./startq.py {}'.format(settings.Q_MANAGER))
    else:
        local('./startq.py {} &'.format(settings.Q_MANAGER))
    if 'LAUNCH_SERVER' in os.environ:
        p = subprocess.Popen(['python','manage.py','runserver','0.0.0.0:8000'])
        p.wait()
    elif 'LAUNCH_SERVER_NGINX' in os.environ:
        subprocess.check_output(["chmod","0777","-R","/tmp"])
        subprocess.check_output(["python","manage.py","collectstatic","--no-input"])
        subprocess.check_output(["chmod","0777","-R","dva/staticfiles/"])
        # subprocess.check_output(["chmod","0777","-R","/root/media/"])
        try:
            subprocess.check_output(["mv","../configs/nginx.conf","/etc/nginx/"])
        except:
            print "warning assuming that the config was already moved"
            pass
        if 'ENABLE_BASICAUTH' in os.environ:
            try:
                subprocess.check_output(["mv","../configs/nginx-app_password.conf","/etc/nginx/sites-available/default"])
            except:
                print "warning assuming that the config was already moved"
                pass
        else:
            try:
                subprocess.check_output(["mv","../configs/nginx-app.conf","/etc/nginx/sites-available/default"])
            except:
                print "warning assuming that the config was already moved"
                pass
        try:
            subprocess.check_output(["mv","../configs/supervisor-app.conf","/etc/supervisor/conf.d/"])
        except:
            print "warning assuming that the config was already moved"
            pass
        p = subprocess.Popen(['supervisord','-n'])
        p.wait()


@task
def init_fs():
    """
    Initialize filesystem by creating directories
    """
    import django, json
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaui.models import ExternalServer
    from dvaapp.models import TrainedModel, Retriever, DVAPQL
    from dvaui.defaults import EXTERNAL_SERVERS
    from dvaapp.processing import DVAPQLProcess
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
    if 'INIT_PROCESS' in os.environ and DVAPQL.objects.count() == 0:
        fname = os.environ.get('INIT_PROCESS')
        p = DVAPQLProcess()
        try:
            jspec = json.load(file(fname))
        except:
            logging.exception("could not load : {}".format(fname))
        else:
            p.create_from_json(jspec)
            p.launch()
