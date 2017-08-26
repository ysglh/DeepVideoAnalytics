import os,logging,time,boto3, glob,subprocess,calendar,sys
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
import json
import random
import gzip
import shutil
from urllib import urlretrieve
from collections import defaultdict
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/fab.log',
                    filemode='a')


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
    """
    Start server locally
    :return:
    """
    local("python manage.py runserver")

@task
def pull_private():
    local('aws s3 cp s3://aub3config/.netrc /root/.netrc')
    local('git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo')
    local('mv DeepVideoAnalyticsDemo dvap')
    local('rm /root/.netrc')
    with lcd('dvap'):
        local('./setup_private.sh')


@task
def start_container_server():
    """
    Start container
    :param test:
    :return:
    """
    local('sleep 20')
    migrate()
    init_fs()
    init_server()
    init_models()
    launch_workers_and_scheduler_from_environment()
    launch_server_from_environment()


@task
def start_container_worker():
    """
    Start worker container
    :param test:
    :return:
    """
    local('sleep 20')
    init_fs()
    init_models()
    launch_workers_and_scheduler_from_environment(block_on_manager=True)

@task
def clean():
    """
    Reset database, migrate, clear media folder, and (only on my dev machine) kill workers/clear all queues.
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp.operations import queuing
    from dvaapp.models import VDNServer
    if sys.platform == 'darwin':
        for qname in set(queuing.TASK_NAMES_TO_QUEUE.values()):
            try:
                local('rabbitmqadmin purge queue name={}'.format(qname))
            except:
                logging.warning("coudnt clear queue {}".format(qname))
    # TODO: wait for Celery bug fix https://github.com/celery/celery/issues/3620
    # local('celery amqp exchange.delete broadcast_tasks')
    migrate()
    local('python manage.py flush --no-input')
    migrate()
    local("rm -rf {}/*".format(settings.MEDIA_ROOT))
    local("mkdir {}/queries".format(settings.MEDIA_ROOT))
    if sys.platform == 'darwin':
        local("rm logs/*.log")
        try:
            local("ps auxww | grep 'celery -A dva worker' | awk '{print $2}' | xargs kill -9")
        except:
            pass
    init_server()
    init_models()
    init_fs()
    if sys.platform == 'darwin':
        superu()

@task
def restart_queues():
    """
    tries to kill all celery workers and restarts them
    :return:
    """
    kill()
    launch()

@task
def kill():
    try:
        local("ps auxww | grep 'celery -A dva worker * ' | awk '{print $2}' | xargs kill -9")
    except:
        pass


@task
def ci():
    """
    Used in conjunction with travis for Continuous Integration testing
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    import base64
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file, pull_vdn_list\
        ,import_vdn_dataset_url
    from dvaapp.models import Video, Clusters,IndexEntries,TEvent,VDNServer, DVAPQL
    from django.conf import settings
    from dvaapp.operations.processing import DVAPQLProcess
    from dvaapp.tasks import perform_dataset_extraction, perform_indexing, perform_export, perform_import,\
        perform_clustering, perform_analysis, perform_detection,\
        perform_video_segmentation, perform_transformation
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name, False)
    if sys.platform != 'darwin':
        for fname in glob.glob('tests/*.mp4'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
            handle_uploaded_file(f, name, False)
        for fname in glob.glob('tests/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
    for i,v in enumerate(Video.objects.all()):
        if v.dataset:
            arguments = {'sync':True}
            perform_dataset_extraction(TEvent.objects.create(video=v,arguments=arguments).pk)
        else:
            arguments = {'sync':True}
            perform_video_segmentation(TEvent.objects.create(video=v,arguments=arguments).pk)
            arguments = {'index': 'inception'}
            perform_indexing(TEvent.objects.create(video=v,arguments=arguments).pk)
        if i ==0: # save travis time by just running detection on first video
            # face_mtcnn
            arguments = {'detector': 'face'}
            dt = TEvent.objects.create(video=v,arguments=arguments)
            perform_detection(dt.pk)
            arguments = {'filters':{'event_id':dt.pk},}
            perform_transformation(TEvent.objects.create(video=v,arguments=arguments).pk)
            # coco_mobilenet
            arguments = {'detector': 'coco'}
            dt = TEvent.objects.create(video=v, arguments=arguments)
            perform_detection(dt.pk)
            arguments = {'filters':{'event_id':dt.pk},}
            perform_transformation(TEvent.objects.create(video=v,arguments=arguments).pk)
            # inception on crops from detector
            arguments = {'index':'inception','target': 'regions','filters': {'event_id': dt.pk, 'w__gte': 50, 'h__gte': 50}}
            perform_indexing(TEvent.objects.create(video=v,arguments=arguments).pk)
            # assign_open_images_text_tags_by_id(TEvent.objects.create(video=v).pk)
        temp = TEvent.objects.create(video=v,arguments={'destination':"FILE"})
        perform_export(temp.pk)
        temp.refresh_from_db()
        fname = temp.arguments['file_name']
        f = SimpleUploadedFile(fname, file("{}/exports/{}".format(settings.MEDIA_ROOT,fname)).read(), content_type="application/zip")
        vimported = handle_uploaded_file(f, fname)
        perform_import(TEvent.objects.create(video=vimported,arguments={"source":"LOCAL"}).pk)
    dc = Clusters()
    dc.indexer_algorithm = 'inception'
    dc.included_index_entries_pk = [k.pk for k in IndexEntries.objects.all().filter(algorithm=dc.indexer_algorithm)]
    dc.components = 32
    dc.save()
    clustering_task = TEvent()
    clustering_task.arguments = {'clusters_id':dc.pk}
    clustering_task.operation = 'perform_clustering'
    clustering_task.save()
    perform_clustering(clustering_task.pk)
    query_dict = {
        'process_type': DVAPQL.QUERY,
        'image_data_b64':base64.encodestring(file('tests/query.png').read()),
        'indexer_queries':[
            {
                'algorithm':'inception',
                'count':10,
                'approximate':False
            }
        ]
    }
    qp = DVAPQLProcess()
    qp.create_from_json(query_dict)
    # execute_index_subquery(qp.indexer_queries[0].pk)
    query_dict = {
        'process_type': DVAPQL.QUERY,
        'image_data_b64':base64.encodestring(file('tests/query.png').read()),
        'indexer_queries':[
            {
                'algorithm':'inception',
                'count':10,
                'approximate':True
            }
        ]
    }
    qp = DVAPQLProcess()
    qp.create_from_json(query_dict)
    # execute_index_subquery(qp.indexer_queries[0].pk)
    server, datasets, detectors = pull_vdn_list(1)
    for k in datasets:
        if k['name'] == 'MSCOCO_Sample_500':
            print 'FOUND MSCOCO SAMPLE'
            import_vdn_dataset_url(VDNServer.objects.get(pk=1),k['url'],None)
    test_backup()


@task
def ci_face():
    """
    Used in conjunction with travis for Continuous Integration for testing face indexing
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video, TEvent
    from dvaapp.tasks import perform_indexing
    for i,v in enumerate(Video.objects.all()):
        if i ==0: # save travis time by just running detection on first video
            args = {
                'filter':{'object_name__startswith':'MTCNN_face'},
                'index':'facenet',
                'target':'regions'}
            perform_indexing(TEvent.objects.create(video=v,arguments=args).pk)


@task
def quick():
    """
    Used on my local Mac for quickly cleaning and testing
    :param detection:
    :return:
    """
    clean()
    superu()
    test()
    launch()


@task
def test_backup():
    """
    Test if backup followed by restore works
    :return:
    """
    local('fab backup')
    clean()
    local('fab restore:backups/*.zip')


@task
def superu():
    """
    Create a superuser
    :return:
    """
    local('echo "from django.contrib.auth.models import User; User.objects.create_superuser(\'akshay\', \'akshay@test.com\', \'super\')" | python manage.py shell')


@task
def launch():
    """
    Launch workers for each queue
    :return:
    """
    envars = ['LAUNCH_BY_NAME_indexer_inception','LAUNCH_BY_NAME_indexer_facenet',
              'LAUNCH_BY_NAME_retriever_inception','LAUNCH_BY_NAME_retriever_facenet',
              'LAUNCH_BY_NAME_detector_coco','LAUNCH_BY_NAME_detector_face',
              'LAUNCH_Q_qclusterer','LAUNCH_Q_qextract']
    for k in envars:
        os.environ[k]="1"
    launch_workers_and_scheduler_from_environment(False)


def launch_workers_and_scheduler_from_environment(block_on_manager=False):
    """
    Launch workers and scheduler as specified in the environment variables.
    Only one scheduler should be launched per deployment.

    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Detector,Indexer,Analyzer
    from dvaapp.operations import queuing
    for k in os.environ:
        if k.startswith('LAUNCH_BY_NAME_'):
            qtype,model_name = k.split('_')[-2:]
            if qtype == 'indexer':
                queue_name = 'q_indexer_{}'.format(Indexer.objects.get(name=model_name).pk)
            elif qtype == 'retriever':
                queue_name = 'q_retriever_{}'.format(Indexer.objects.get(name=model_name).pk)
            elif qtype == 'detector':
                queue_name = 'q_detector_{}'.format(Detector.objects.get(name=model_name).pk)
            elif qtype == 'analyzer':
                queue_name = 'q_analyzer_{}'.format(Analyzer.objects.get(name=model_name).pk)
            else:
                raise ValueError,k
            command = 'fab startq:{} &'.format(queue_name)
            logging.info("'{}' for {}".format(command,k))
            local(command)
        elif k.startswith('LAUNCH_Q_') and k != 'LAUNCH_Q_{}'.format(queuing.Q_MANAGER):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                local('fab startq:{},{} &'.format(queue_name,os.environ['LAUNCH_Q_qextract']))
            else:
                queue_name = k.split('_')[-1]
                local('fab startq:{} &'.format(queue_name))
    if os.environ.get("LAUNCH_SCHEDULER",False):
        # Should be launched only once per deployment
        local('fab start_scheduler &')
    if block_on_manager: # the container process waits on the manager
        local('fab startq:{}'.format(queuing.Q_MANAGER))
    else:
        local('fab startq:{} &'.format(queuing.Q_MANAGER))


def launch_server_from_environment():
    """
    Launch django development server or NGINX server.
    :return:
    """
    if 'LAUNCH_SERVER' in os.environ:
        local('python manage.py runserver 0.0.0.0:8000')
    elif 'LAUNCH_SERVER_NGINX' in os.environ:
        local('chmod 0777 -R /tmp')
        try:
            local("mv docker/configs/nginx.conf /etc/nginx/")
        except:
            print "warning assuming that the config was already moved"
            pass
        if 'ENABLE_BASICAUTH' in os.environ:
            try:
                local("mv docker/configs/nginx-app_password.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        else:
            try:
                local("mv docker/configs/nginx-app.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        try:
            local("mv docker/configs/supervisor-app.conf /etc/supervisor/conf.d/")
        except:
            print "warning assuming that the config was already moved"
            pass
        local("python manage.py collectstatic --no-input")
        local("chmod 0777 -R dva/staticfiles/")
        # local("chmod 0777 -R dva/media/")
        local('supervisord -n')



@task
def download_indexers(root_dir):
    indexer_dir = "{}/indexers/".format(root_dir)
    if not os.path.isdir(indexer_dir):
        os.mkdir(indexer_dir)
    with lcd(indexer_dir):
        ilist = [('facenet','https://www.dropbox.com/s/jytpgw8et09ede9/facenet.pb','facenet.pb'),
                 ('vgg', 'https://www.dropbox.com/s/3yzonc9nzo9xanv/vgg.pb', 'vgg.pb'),
                 ('inception', 'https://www.dropbox.com/s/fc7li2vwn8lvsyu/network.pb', 'network.pb'),
                 ]
        for iname, iurl, lfname in ilist:
            model_dir = "{}/indexers/{}".format(root_dir, iname)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
                if sys.platform == 'darwin':
                    local("cd {} && cp /users/aub3/Dropbox/DeepVideoAnalytics/shared/{} .".format(iname,lfname))
                else:
                    local("cd {} && wget --quiet {}".format(iname,iurl))


@task
def download_detectors(root_dir):
    detectors_dir = "{}/detectors/".format(root_dir)
    if not os.path.isdir(detectors_dir):
        os.mkdir(detectors_dir)
    with lcd(detectors_dir):
        ilist = [('coco','https://www.dropbox.com/s/nzz26b2p4wxygg3/coco_mobilenet.pb','coco_mobilenet.pb'),
                 ('yolo', 'https://www.dropbox.com/s/zbff2rkoejx5k5r/yolo.h5', 'yolo.h5'),]
        for iname, url, lfname in ilist:
            model_dir = "{}/detectors/{}".format(root_dir,iname)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
                if sys.platform == 'darwin':
                    local("cd {} && cp /users/aub3/Dropbox/DeepVideoAnalytics/shared/{} .".format(iname,lfname))
                else:
                    local("cd {} && wget --quiet {}".format(iname,url))



@task
def download_analyzers(root_dir):
    analyzers_dir = "{}/analyzers/".format(root_dir)
    if not os.path.isdir(analyzers_dir):
        os.mkdir(analyzers_dir)
    with lcd(analyzers_dir):
        ilist = [('crnn','https://www.dropbox.com/s/l0vo83hmvv2aipn/crnn.pth','crnn.pth'),]
        for iname, url, lfname in ilist:
            model_dir = "{}/analyzers/{}".format(root_dir,iname)
            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)
                if sys.platform == 'darwin':
                    local("cd {} && cp /users/aub3/Dropbox/DeepVideoAnalytics/shared/{} .".format(iname,lfname))
                else:
                    local("cd {} && wget --quiet {}".format(iname,url))



@task
def init_server():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video,VDNServer,StoredDVAPQL
    if StoredDVAPQL.objects.count() == 0:
        for fname in glob.glob('dvaapp/test_scripts/*.json'):
            StoredDVAPQL.objects.create(name=fname,
                                        process_type=StoredDVAPQL.PROCESS,
                                        script=json.loads(file(fname).read()))
    if not ('DISABLE_VDN' in os.environ):
        if VDNServer.objects.count() == 0:
            server = VDNServer()
            server.url = "http://www.visualdata.network/"
            server.name = "VisualData.Network"
            server.save()
    if 'TEST' in os.environ and Video.objects.count() == 0:
        test()


@task
def init_models():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Detector,Analyzer,Indexer
    _ = Detector.objects.get_or_create(name="coco",algorithm="mobilenet_ssd")
    _ = Detector.objects.get_or_create(name="face",algorithm="mtcnn_facenet")
    _ = Detector.objects.get_or_create(name="textbox",algorithm="cptn")
    _ = Indexer.objects.get_or_create(name="inception")
    _ = Indexer.objects.get_or_create(name="facenet")
    _ = Analyzer.objects.get_or_create(name="tag")
    _ = Analyzer.objects.get_or_create(name="crnn")


@task
def init_fs():
    """
    Initialized filesystem by downloading models
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    download_indexers(settings.MEDIA_ROOT)
    download_detectors(settings.MEDIA_ROOT)
    download_analyzers(settings.MEDIA_ROOT)


@task
def add_default_vdn_server():
    """
    Add http://www.visualdata.network/ as default VDN server
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video,VDNServer
    if VDNServer.objects.count() == 0:
        server = VDNServer()
        server.url = "http://www.visualdata.network/"
        server.name = "VisualData.Network"
        server.save()


@task
def startq(queue_name,conc=3):
    """
    Start worker to handle a queue, Usage: fab startq:indexer
    Concurrency is set to 1 but you can edit code to change.
    :param queue_name: indexer, extractor, retriever, detector
    :param conc:conccurency only for extractor
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.operations import queuing
    mute = '--without-gossip --without-mingle --without-heartbeat' if 'CELERY_MUTE' in os.environ else ''
    if queue_name == queuing.Q_MANAGER:
        command = 'celery -A dva worker -l info {} -c 1 -Q qmanager -n manager.%h -f logs/qmanager.log'.format(mute)
    elif queue_name == queuing.Q_EXTRACTOR:
        command = 'celery -A dva worker -l info {} -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute,max(int(conc),2), queue_name,queue_name,queue_name)
        # TODO: worker fails due to
        # https://github.com/celery/celery/issues/3620
    else:
        command = 'celery -A dva worker -l info {} -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute,1, queue_name,queue_name,queue_name)
    logging.info(command)
    os.system(command)



@task
def test():
    """
    Run tests by launching tasks
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file, handle_video_url
    for fname in glob.glob('tests/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)


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


def setup_django():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()


@task
def heroku_migrate():
    local('heroku run python manage.py migrate')


# @task
# def heroku_update_env():
#     local('heroku config:get DATABASE_URL > db.env')


@task
def heroku_shell():
    local('heroku run python manage.py shell')


@task
def heroku_bash():
    local('heroku run bash')


@task
def heroku_config():
    local('heroku config')


@task
def heroku_psql():
    local('heroku pg:psql')


@task
def heroku_reset(password,bucket_name):
    if raw_input("Are you sure type yes >>") == 'yes':
        local('heroku pg:reset DATABASE_URL')
        heroku_migrate()
        heroku_setup_vdn(password)
        local('heroku run python manage.py createsuperuser')
        print "emptying bucket"
        local("aws s3 rm s3://{} --recursive --quiet".format(bucket_name))


@task
def heroku_super():
    local('heroku run python manage.py createsuperuser')


@task
def heroku_local_static():
    local('python manage.py collectstatic')


@task
def heroku_migrate():
    local('heroku run python manage.py migrate')


@task
def heroku_setup_vdn(password):
    local('heroku run fab setup_vdn:{}'.format(password))


@task
def make_requester_pays(bucket_name):
    """
    Convert AWS S3 bucket into requester pays bucket
    DOES NOT WORKS,
    :param bucket_name:
    :return:
    """
    s3 = boto3.resource('s3')
    bucket_request_payment = s3.BucketRequestPayment(bucket_name)
    response = bucket_request_payment.put(RequestPaymentConfiguration={'Payer': 'Requester'})
    bucket_policy = s3.BucketPolicy(bucket_name)
    policy = {
          "Id": "Policy1493037034955",
          "Version": "2012-10-17",
          "Statement": [
            {
              "Sid": "Stmt1493036947566",
              "Action": [
                "s3:ListBucket"
              ],
              "Effect": "Allow",
              "Resource": "arn:aws:s3:::{}".format(bucket_name),
              "Principal": "*"
            },
            {
              "Sid": "Stmt1493037029723",
              "Action": [
                "s3:GetObject"
              ],
              "Effect": "Allow",
              "Resource": "arn:aws:s3:::{}/*".format(bucket_name),
              "Principal": {
                "AWS": [
                  "*"
                ]
              }
            }
          ]}
    response = bucket_policy.put(Policy=json.dumps(policy))



@task
def setup_vdn(password):
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from vdnapp.models import User,Dataset,Organization,VDNRemoteDetector
    user = User(username="akshayvdn",password=password,email="aub3@cornell.edu")
    user.save()
    o = Organization()
    o.user = user
    o.description = "Default organization"
    o.name = "Akshay Bhat"
    o.save()
    Dataset.objects.all().delete()
    datasets = [
        ('LFW_subset','https://www.dropbox.com/s/6nn84z4yzy47vuh/LFW.dva_export.zip'),
        ('MSCOCO_Sample_500','https://www.dropbox.com/s/qhzl9ig7yhems6j/MSCOCO_Sample.dva_export.zip'),
        ('Paris','https://www.dropbox.com/s/a7qf1f3j8vp4nuh/Paris.dva_export.zip'),
    ]
    for name,url in datasets:
        d = Dataset()
        d.organization = o
        d.download_url = url
        d.name = name
        d.root = True
        d.aws_requester_pays = False
        d.description = name
        d.save()
    aws_datasets = [
        ('MSCOCO train ~14GB', 'us-east-1','visualdatanetwork','coco_train.dva_export.zip'),
        ('aws_test_dir', 'us-east-1', 'visualdatanetwork', '007'),
    ]
    for name, region, bucket, key in aws_datasets:
        d = Dataset()
        d.organization = o
        d.name = name
        d.aws_region = region
        d.aws_bucket = bucket
        d.aws_key = key
        d.root = True
        d.aws_requester_pays = True
        d.description = name
        d.save()
    detectors = [
        ('License plate', 'https://www.dropbox.com/s/ztsl59pxgzvd14k/1.dva_detector.zip'),
    ]
    for name, url in detectors:
        d = VDNRemoteDetector()
        d.organization = o
        d.name = name
        d.download_url = url
        d.description = name
        d.save()


@task
def heroku_setup():
    local('heroku buildpacks:add https://github.com/AKSHAYUBHAT/heroku-buildpack-run.git')
    local('heroku config:set DISABLE_COLLECTSTATIC=1')


@task
def sync_static(bucket_name='dvastatic'):
    local_static()
    with lcd('dva'):
        local('aws s3 sync staticfiles/ s3://{}/'.format(bucket_name))


@task
def enable_media_bucket_static_hosting(bucket_name, allow_videos=False):
    """
    Enable static hosting for given bucket name
    Note that the bucket / media becomes publicly viewable.
    An alternative is using presigned url but it will require a django filter
    https://stackoverflow.com/questions/33549254/how-to-generate-url-from-boto3-in-amazon-web-services
    :param bucket_name:
    :return:
    """
    s3 = boto3.client('s3')
    cors_configuration = {
        'CORSRules': [{
            'AllowedHeaders': ['*'],
            'AllowedMethods': ['GET'],
            'AllowedOrigins': ['*'],
            'ExposeHeaders': ['GET'],
            'MaxAgeSeconds': 3000
        }]
    }
    s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_configuration)
    bucket_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Sid': 'AddPerm',
            'Effect': 'Allow',
            'Principal': '*',
            'Action': ['s3:GetObject'],
            'Resource': "arn:aws:s3:::%s/*.jpg" % bucket_name
        },
            {
            'Sid': 'AddPerm',
            'Effect': 'Allow',
            'Principal': '*',
            'Action': ['s3:GetObject'],
            'Resource': "arn:aws:s3:::%s/*.png" % bucket_name
        }]
    }
    if allow_videos:
        bucket_policy['Statement'].append({
                'Sid': 'AddPerm',
                'Effect': 'Allow',
                'Principal': '*',
                'Action': ['s3:GetObject'],
                'Resource': "arn:aws:s3:::%s/*.mp4" % bucket_name
            })
    bucket_policy = json.dumps(bucket_policy)
    s3.put_bucket_policy(Bucket=bucket_name, Policy=bucket_policy)
    website_configuration = {'ErrorDocument': {'Key': 'error.html'},'IndexDocument': {'Suffix': 'index.html'},}
    s3.put_bucket_website(Bucket=bucket_name,WebsiteConfiguration=website_configuration)


@task
def sync_efs_to_s3():
    setup_django()
    from dvaapp.models import Video,TEvent
    from dvaapp.tasks import perform_sync
    for v in Video.objects.all():
        e = TEvent()
        e.video_id = v.pk
        e.operation = 'perform_sync'
        e.save()
        perform_sync(e.pk)


@task
def train_yolo(start_pk):
    """
    Train a yolo model specified in a TaskEvent.
    This is necessary to ensure that the Tensorflow process exits and releases the allocated GPU memory.
    :param start_pk: TEvent PK with information about lauching the training task
    :return:
    """
    setup_django()
    from django.conf import settings
    from dvaapp.models import Region, Frame, Detector, TEvent
    from dvaapp.shared import create_detector_dataset
    from dvalib.yolo import trainer
    start = TEvent.objects.get(pk=start_pk)
    args = start.arguments
    labels = set(args['labels']) if 'labels' in args else set()
    object_names = set(args['object_names']) if 'object_names' in args else set()
    detector = Detector.objects.get(pk=args['detector_pk'])
    detector.create_directory()
    args['root_dir'] = "{}/detectors/{}/".format(settings.MEDIA_ROOT,detector.pk)
    args['base_model'] = "{}/detectors/yolo/yolo.h5"
    class_distribution, class_names, rboxes, rboxes_set, frames, i_class_names = create_detector_dataset(object_names,labels)
    images, boxes = [], []
    path_to_f = {}
    for k,f in frames.iteritems():
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,f.video_id,f.frame_index)
        path_to_f[path] = f
        images.append(path)
        boxes.append(rboxes[k])
        # print k,rboxes[k]
    with open("{}/input.json".format(args['root_dir']),'w') as input_data:
        json.dump({'boxes':boxes,
                   'images':images,
                   'args':args,
                   'class_names':class_names.items(),
                   'class_distribution':class_distribution.items()},
                  input_data)
    detector.boxes_count = sum([len(k) for k in boxes])
    detector.frames_count = len(images)
    detector.classes_count = len(class_names)
    detector.save()
    train_task = trainer.YOLOTrainer(boxes=boxes,images=images,class_names=i_class_names,args=args)
    train_task.train()
    detector.phase_1_log = file("{}/phase_1.log".format(args['root_dir'])).read()
    detector.phase_2_log = file("{}/phase_2.log".format(args['root_dir'])).read()
    detector.class_distribution = json.dumps(class_distribution.items())
    detector.class_names = json.dumps(class_names.items())
    detector.trained = True
    detector.save()
    results = train_task.predict()
    bulk_regions = []
    for path, box_class, score, top, left, bottom, right in results:
        r = Region()
        r.region_type = r.ANNOTATION
        r.confidence = int(100.0 * score)
        r.object_name = "YOLO_{}_{}".format(detector.pk,box_class)
        r.y = top
        r.x = left
        r.w = right - left
        r.h = bottom - top
        r.frame_id = path_to_f[path].pk
        r.video_id = path_to_f[path].video_id
        bulk_regions.append(r)
    Region.objects.bulk_create(bulk_regions,batch_size=1000)
    folder_name = "{}/detectors/{}".format(settings.MEDIA_ROOT,detector.pk)
    file_name = '{}/exports/{}.dva_detector.zip'.format(settings.MEDIA_ROOT,detector.pk)
    zipper = subprocess.Popen(['zip', file_name, '-r', '.'],cwd=folder_name)
    zipper.wait()
    return 0


@task
def temp_import_detector(path="/Users/aub3/tempd"):
    """
    For testing pre-developed detectors
    :param path:
    :return:
    """
    setup_django()
    import json
    from django.conf import settings
    from dvaapp.models import Detector
    d = Detector()
    with open("{}/input.json".format(path)) as infile:
        data = json.load(infile)
    d.name = "test detector"
    d.class_names = json.dumps(data['class_names'])
    d.phase_1_log = file("{}/phase_1.log".format(path)).read
    d.phase_2_log = file("{}/phase_2.log".format(path)).read
    d.frames_count = 500
    d.boxes_count = 500
    d.class_distribution = json.dumps(data['class_names'])
    d.save()
    d.create_directory()
    shutil.copy("{}/phase_2_best.h5".format(path),"{}/detectors/{}/phase_2_best.h5".format(settings.MEDIA_ROOT,d.pk))


@task
def qt():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/mp4")
        v = handle_uploaded_file(f, name)
    for fname in glob.glob('tests/example*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        v = handle_uploaded_file(f, name)


@task
def create_custom_migrations():
    """
    Create custom migration files for adding default indexers (inception_v3, facenet)
    and postgres text search indexes for fulltext search

    To reset postgres on dev machine use "drop schema public cascade;create schema public;"
    :return:
    """
    local('python manage.py makemigrations --empty --name textsearch_indexes dvaapp')
    local('python manage.py makemigrations --empty --name default_indexers dvaapp')


@task
def submit(path):
    """
    Submit a DVAPQL process to run
    :param path:
    :return: id of the submitted process
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.operations.processing import DVAPQLProcess
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
    :return:
    """
    local("celery -A dva beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler -f logs/beat.log")


@task
def store_token_for_testing():
    """
    Generate & store token for superuser (akshay) to test REST API.
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.contrib.auth.models import User
    from rest_framework.authtoken.models import Token
    try:
        u = User.objects.get(username="akshay")
    except:
        superu()
        u = User.objects.get(username="akshay")
    token, _ = Token.objects.get_or_create(user=User.objects.get(username=u))
    with open('creds.json','w') as creds:
        creds.write(json.dumps({'token':token.key}))


@task
def test_api(port=80):
    """
    test REST API for CORS config
    :return:
    """
    import requests
    if not os.path.isfile('creds.json'):
        store_token_for_testing()
    token = json.loads(file('creds.json').read())['token']
    headers={'Authorization':'Token {}'.format(token)}
    r = requests.post("http://localhost:{}/api/queries/".format(port),
                      data={'script':file('dvaapp/test_scripts/url.json').read()},
                      headers=headers)
    print r.status_code


@task
def capture_stream(url="https://www.youtube.com/watch?v=vpm16w3ik0g"):
    command = 'livestreamer --player-continuous-http --player-no-close ' \
              '"{}" best -O --yes-run-as-root | ' \
              'ffmpeg -re -i - -c:v libx264 -c:a aac -ac 1 -strict -2 -crf 18 ' \
              '-profile:v baseline -maxrate 3000k -bufsize 1835k -pix_fmt yuv420p ' \
              '-flags -global_header -f segment -segment_time 0.1 "%d.mp4"'.format(url)
    if raw_input("This code uses os.system and is a huge security risk if url is malicious shell string. Type yes to confirm>>") == "yes":
        print command
        os.system(command)