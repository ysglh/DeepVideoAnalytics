import os,logging,time,boto3, glob,subprocess,calendar,sys
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/fab.log',
                    filemode='a')
AMI = 'ami-b3cc1fa5'
KeyName = 'C:\\fakepath\\cs5356'
SecurityGroupId = 'sg-06a6b562'
IAM_ROLE = {'Arn': 'arn:aws:iam::248089713624:instance-profile/chdeploy',}
env.user = "ubuntu"
try:
    ec2_HOST = file("host").read().strip()
except:
    ec2_HOST = ""
    pass
env.key_filename = "~/.ssh/cs5356"


def get_status(ec2,spot_request_id):
    """
    Get status of EC2 spot request
    :param ec2:
    :param spot_request_id:
    :return:
    """
    current = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id,])
    instance_id = current[u'SpotInstanceRequests'][0][u'InstanceId'] if u'InstanceId' in current[u'SpotInstanceRequests'][0] else None
    return instance_id


@task
def launch_spot():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    ec2spec = dict(ImageId=AMI,
                   KeyName = KeyName,
                   SecurityGroupIds = [SecurityGroupId, ],
                   InstanceType = "p2.xlarge",
                   Monitoring = {'Enabled': True,},
                   IamInstanceProfile = IAM_ROLE)
    output = ec2.request_spot_instances(DryRun=False,
                                        SpotPrice="0.9",
                                        InstanceCount=1,
                                        LaunchSpecification = ec2spec)
    spot_request_id = output[u'SpotInstanceRequests'][0][u'SpotInstanceRequestId']
    time.sleep(30)
    waiter = ec2.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[spot_request_id,])
    instance_id = get_status(ec2, spot_request_id)
    while instance_id is None:
        time.sleep(30)
        instance_id = get_status(ec2,spot_request_id)
    instance = ec2r.Instance(instance_id)
    with open("host",'w') as out:
        out.write(instance.public_ip_address)
    time.sleep(12) # wait while the instance starts
    env.hosts = [instance.public_ip_address,]
    fh = open("connect.sh", 'w')
    fh.write("#!/bin/bash\n" + "ssh -i " + env.key_filename + " " + env.user + "@" + env.hosts[0] + "\n")
    fh.close()
    local("fab deploy_ec2")

@task
def deploy_ec2():
    """
    deploys code on hostname
    :return:
    """
    import webbrowser
    run('cd deepvideoanalytics && git pull && cd docker_GPU && ./rebuild.sh && nvidia-docker-compose up -d')
    webbrowser.open('{}:8000'.format(env.hosts[0]))


@task
def shell():
    """
    start a local django shell
    :return:
    """
    local('python manage.py shell')



@task
def local_static():
    """
    Collect static
    :return:
    """
    local('python manage.py collectstatic')


@task
def migrate():
    """
    Make migrations and migrate database
    :return:
    """
    local('python manage.py makemigrations')
    local('python manage.py migrate')




@task
def server():
    local("python manage.py runserver")


@task
def start_server_container(test=False):
    local('sleep 60')
    migrate()
    launch_queues(True)
    if test:
        ci()
    local('python manage.py runserver 0.0.0.0:8000')


@task
def start_server_container_gpu(test=False):
    local('sleep 60')
    migrate()
    local('chmod 0777 -R /tmp')
    local("mv docker_GPU/configs/nginx.conf /etc/nginx/")
    local("mv docker_GPU/configs/nginx-app.conf /etc/nginx/sites-available/default")
    local("mv docker_GPU/configs/supervisor-app.conf /etc/supervisor/conf.d/")
    local("python manage.py collectstatic --no-input")
    local("chmod 0777 -R dva/staticfiles/")
    local("chmod 0777 -R dva/media/")
    launch_queues(True)
    if test:
        ci()
    local('supervisord -n')


@task
def clean():
    for qname in ['qextract','qindexer','qdetector','qretriever']:
        try:
            local('rabbitmqadmin purge queue name={}'.format(qname))
        except:
            logging.warning("coudnt clear queue {}".format(qname))
    local('python manage.py makemigrations')
    local('python manage.py flush --no-input')
    migrate()
    local("rm logs/*.log")
    local("rm -rf ~/media/*")
    local("mkdir ~/media/queries")
    try:
        local("ps auxww | grep 'celery -A dva worker' | awk '{print $2}' | xargs kill -9")
    except:
        pass


@task
def ci():
    """
    Used in conjunction with travis for Continuous Integration testing
    :return:
    """
    with lcd("tests"):
        local('wget https://www.dropbox.com/s/xwbk5g1qit5s9em/WorldIsNotEnough.mp4')
        local('wget https://www.dropbox.com/s/misaejsbz6722pd/test.png')
    test(True)


@task
def quick_test(detection=False):
    """
    Used on my local Mac for quickly cleaning and testing
    :param detection:
    :return:
    """
    clean()
    create_super()
    test()  # test without launch tasks
    launch_queues(detection)


@task
def test_backup():
    local('fab backup')
    clean()
    local('fab restore:backups/*.zip')


@task
def create_super():
    local('echo "from django.contrib.auth.models import User; User.objects.create_superuser(\'akshay\', \'akshay@test.com\', \'super\')" | python manage.py shell')


@task
def launch_queues(detection=False):
    local('fab startq:extractor &')
    local('fab startq:indexer &')
    local('fab startq:retriever &')
    if detection:
        local('fab startq:detector &')


@task
def startq(queue_name):
    """
    Start worker to handle a queue, Usage: fab startq:indexer
    :param queue_name: indexer, extractor, retriever, detector
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    Q_INDEXER = settings.Q_INDEXER
    Q_EXTRACTOR = settings.Q_EXTRACTOR
    Q_DETECTOR = settings.Q_DETECTOR
    Q_RETRIEVER = settings.Q_RETRIEVER
    if queue_name == 'indexer':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_INDEXER, Q_INDEXER,Q_INDEXER)
    elif queue_name == 'extractor':
        if len(sys.argv) > 3:
            concurrency = int(sys.argv[3])
        else:
            concurrency = 1
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(concurrency,Q_EXTRACTOR,Q_EXTRACTOR,Q_EXTRACTOR)
    elif queue_name == 'detector':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_DETECTOR,Q_DETECTOR, Q_DETECTOR)
    elif queue_name == 'retriever':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_RETRIEVER,Q_RETRIEVER,Q_RETRIEVER)
    else:
        raise NotImplementedError
    logging.info(command)
    os.system(command)


@task
def test(ci=False):
    """
    Run tests
    :param ci: if True (fab test:1) tests are run on Travis this option skips creating tasks and directly calls
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file, handle_youtube_video
    from dvaapp.models import Video
    from dvaapp.tasks import extract_frames, perform_indexing, perform_detection
    if ci:
        for fname in glob.glob('tests/ci/*.mp4'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
            handle_uploaded_file(f, name, False)
        for fname in glob.glob('tests/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
        handle_youtube_video('jungle book', 'https://www.youtube.com/watch?v=C4qgAaxB_pc')
        for v in Video.objects.all():
            extract_frames(v.pk)
            perform_indexing(v.pk)
            perform_detection(v.pk)
        test_backup()
    else:
        for fname in glob.glob('tests/*.mp4'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
            handle_uploaded_file(f, name)
        for fname in glob.glob('tests/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
        handle_youtube_video('jungle book', 'https://www.youtube.com/watch?v=C4qgAaxB_pc')


@task
def backup():
    """
    Take a backup, backups are store as a single zip file in backups/ folder
    :return:
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
    :return:
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
            [pg, '--dbname', 'postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'], db['NAME'])],
            cwd=media_dir, stdin=dumpfile)
        dump.communicate()
    print dump.returncode
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode
