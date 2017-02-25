import sys,dva,os
# Tasks running on GPU should have concurrency set to 1 otherwise, it might prosent an issue with GPU Memory allocation

if __name__ == '__main__':
    Q_INDEXER = dva.settings.Q_INDEXER
    Q_EXTRACTOR = dva.settings.Q_EXTRACTOR
    Q_DETECTOR = dva.settings.Q_DETECTOR
    if sys.argv[1] == 'indexer':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1,Q_INDEXER,Q_INDEXER,Q_INDEXER)
    elif sys.argv[1] == 'extractor':
        if len(sys.argv) > 2:
            concurrency = int(sys.argv[2])
        else:
            concurrency = 1
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(concurrency, Q_EXTRACTOR,Q_EXTRACTOR, Q_EXTRACTOR)
    elif sys.argv[1] == 'detector':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_DETECTOR, Q_DETECTOR,Q_DETECTOR)
    else:
        raise NotImplementedError
    print command
    os.system(command)

