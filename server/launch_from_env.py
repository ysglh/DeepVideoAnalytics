#!/usr/bin/env python
import django, os, sys, logging, subprocess, shlex
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.models import TrainedModel, Retriever
from django.conf import settings

if __name__ == '__main__':
    block_on_manager = False
    if sys.argv[-1] == '1':
            block_on_manager = True
    for k in os.environ:
        if k.startswith('LAUNCH_BY_NAME_'):
            qtype, model_name = k.split('_')[-2:]
            env_vars = ""
            if qtype == 'indexer':
                dm = TrainedModel.objects.get(name=model_name, model_type=TrainedModel.INDEXER)
                queue_name = 'q_indexer_{}'.format(dm.pk)
            elif qtype == 'retriever':
                dm = Retriever.objects.get(name=model_name)
                queue_name = 'q_retriever_{}'.format(dm.pk)
            elif qtype == 'detector':
                dm = TrainedModel.objects.get(name=model_name, model_type=TrainedModel.DETECTOR)
                queue_name = 'q_detector_{}'.format(dm.pk)
            elif qtype == 'analyzer':
                dm = TrainedModel.objects.get(name=model_name, model_type=TrainedModel.ANALYZER)
                queue_name = 'q_analyzer_{}'.format(dm.pk)
            else:
                raise ValueError, k
            envs = os.environ.copy()
            if qtype != 'retriever':
                if dm.mode == dm.PYTORCH:
                    env_vars = "PYTORCH_MODE"
                elif dm.mode == dm.CAFFE:
                    env_vars = "CAFFE_MODE"
                elif dm.mode == dm.MXNET:
                    env_vars = "MXNET_MODE"
                else:
                    env_vars = None
                if env_vars:
                    envs[env_vars] = 1
            _ = subprocess.Popen(['./startq.py',queue_name], env=envs)
        elif k.startswith('LAUNCH_Q_') and k != 'LAUNCH_Q_{}'.format(settings.Q_MANAGER):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                _ = subprocess.Popen(
                    shlex.split(('./startq.py {} {}'.format(queue_name, os.environ['LAUNCH_Q_qextract']))))
            elif k.startswith('LAUNCH_Q_qglobal'):
                queue_name = k.strip('LAUNCH_Q_')
                _ = subprocess.Popen(shlex.split(('./startq.py {}'.format(queue_name))))
            else:
                queue_name = k.split('_')[-1]
                _ = subprocess.Popen(shlex.split(('./startq.py {}'.format(queue_name))))
    if os.environ.get("LAUNCH_SCHEDULER", False):
        # Should be launched only once per deployment
        _ = subprocess.Popen(['./start_scheduler.py'])
    if block_on_manager:  # the container process waits on the manager
        subprocess.check_call('./startq.py {}'.format(settings.Q_MANAGER))
    else:
        _ = subprocess.Popen(shlex.split('./startq.py {}'.format(settings.Q_MANAGER)))
    if 'LAUNCH_SERVER' in os.environ:
        p = subprocess.Popen(['python', 'manage.py', 'runserver', '0.0.0.0:8000'])
        p.wait()
    elif 'LAUNCH_SERVER_NGINX' in os.environ:
        subprocess.check_output(["chmod", "0777", "-R", "/tmp"])
        subprocess.check_output(["python", "manage.py", "collectstatic", "--no-input"])
        subprocess.check_output(["chmod", "0777", "-R", "dva/staticfiles/"])
        # subprocess.check_output(["chmod","0777","-R","/root/media/"])
        try:
            subprocess.check_output(["mv", "../configs/nginx.conf", "/etc/nginx/"])
        except:
            print "warning assuming that the config was already moved"
            pass
        if 'ENABLE_BASICAUTH' in os.environ:
            try:
                subprocess.check_output(["mv", "../configs/nginx-app_password.conf", "/etc/nginx/sites-available/default"])
            except:
                print "warning assuming that the config was already moved"
                pass
        else:
            try:
                subprocess.check_output(["mv", "../configs/nginx-app.conf", "/etc/nginx/sites-available/default"])
            except:
                print "warning assuming that the config was already moved"
                pass
        try:
            subprocess.check_output(["mv", "../configs/supervisor-app.conf", "/etc/supervisor/conf.d/"])
        except:
            print "warning assuming that the config was already moved"
            pass
        p = subprocess.Popen(['supervisord', '-n'])
        p.wait()
