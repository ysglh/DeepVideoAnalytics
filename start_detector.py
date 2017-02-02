import sys,dva,os

if __name__ == '__main__':
    import dva
    Q_DETECTOR = dva.settings.Q_DETECTOR
    command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1,Q_DETECTOR,Q_DETECTOR,Q_DETECTOR)
    print command
    os.system(command)
