#!/usr/bin/env python
import django, os, sys, subprocess, shlex, shutil, glob
sys.path.append('../server/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
import utils


if __name__ == '__main__':
    for qname in set(settings.TASK_NAMES_TO_QUEUE.values()):
        try:
            subprocess.check_output(shlex.split('rabbitmqadmin purge queue name={}'.format(qname)))
        except:
            print "error clearly {} passing".format(qname)
    # TODO: wait for Celery bug fix https://github.com/celery/celery/issues/3620
    # local('celery amqp exchange.delete broadcast_tasks')
    utils.migrate()
    subprocess.check_output(shlex.split('python manage.py flush --no-input'),cwd='../server')
    utils.migrate()
    shutil.copy("../configs/custom_defaults/defaults_mac.py", '../server/dvaui/defaults.py')
    for dirname in os.listdir(settings.MEDIA_ROOT):
        shutil.rmtree("{}/{}".format(settings.MEDIA_ROOT,dirname))
    for log_filename in glob.glob("../logs/*.log"):
        os.remove(log_filename)
    try:
        subprocess.check_output(['./kill.sh'])
    except:
        pass
    envs = os.environ.copy()
    envs['SUPERUSER'] = 'akshay'
    envs['SUPERPASS'] = 'super'
    envs['SUPEREMAIL'] = 'test@deepvideoanalytics.com'
    subprocess.check_output(shlex.split('fab copy_defaults'),cwd='../server',env=envs)
    subprocess.check_output(shlex.split('fab init_fs'),cwd='../server',env=envs)