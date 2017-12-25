#!/usr/bin/env python
import django, os, sys, subprocess, shlex, shutil
sys.path.append('../server/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
import utils


if __name__ == '__main__':
    for qname in set(settings.TASK_NAMES_TO_QUEUE.values()):
        subprocess.check_output(shlex.split('rabbitmqadmin purge queue name={}'.format(qname)))
    # TODO: wait for Celery bug fix https://github.com/celery/celery/issues/3620
    # local('celery amqp exchange.delete broadcast_tasks')
    utils.migrate()
    subprocess.check_output(shlex.split('python manage.py flush --no-input'),cwd='../server')
    utils.migrate()
    shutil.copy("../configs/custom_defaults/defaults_mac.py", 'dvaui/defaults.py')
    for dirname in os.listdir(settings.MEDIA_ROOT):
        if dirname != 'gitkeep':
            shutil.rmtree("rm -rf {}/{}".format(settings.MEDIA_ROOT,dirname))
    os.system("rm ../logs/*.log")
    try:
        os.system("ps auxww | grep 'celery -A dva' | awk '{print $2}' | xargs kill -9")
    except:
        pass
    os.environ['SUPERUSER'] = 'akshay'
    os.environ['SUPERPASS'] = 'super'
    os.environ['SUPEREMAIL'] = 'test@deepvideoanalytics.com'
    subprocess.check_output(shlex.split('fab copy_defaults'),cwd='../server')
    subprocess.check_output(shlex.split('fab init_fs'),cwd='../server')