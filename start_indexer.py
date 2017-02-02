import sys,dva,os

if __name__ == '__main__':
    import dva
    Q_INDEXER = dva.settings.Q_INDEXER
    # Tasks running on GPU should have concurrency set to 1 otherwise, it might prosent an issue with GPU Memory allocation
    command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1,Q_INDEXER,Q_INDEXER,Q_INDEXER)
    print command
    os.system(command)
