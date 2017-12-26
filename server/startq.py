#!/usr/bin/env python
import django, os, subprocess, sys, logging, shlex
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    queue_name = sys.argv[1]
    if len(sys.argv) > 2:
        conc = int(sys.argv[2])
    else:
        conc = 3
    mute = '--without-gossip --without-mingle --without-heartbeat' if 'CELERY_MUTE' in os.environ else ''
    if queue_name == settings.Q_MANAGER:
        command = 'celery -A dva worker -l info {} -c 1 -Q qmanager -n manager.%h -f ../logs/qmanager.log'.format(mute)
    elif queue_name == settings.Q_EXTRACTOR:
        try:
            subprocess.check_output(['youtube-dl', '-U'])
        except:
            logging.exception("Could not update youtube-dl")
            pass
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
