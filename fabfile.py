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
def start_container():
    """
    Start container
    :param test:
    :return:
    """
    local('sleep 20')
    migrate()
    launch_queues_env()
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
    from dvaapp.models import VDNServer
    if sys.platform == 'darwin':
        for qname in set(settings.TASK_NAMES_TO_QUEUE.values()):
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
    server = VDNServer()
    server.url = "http://www.visualdata.network/"
    server.name = "VisualData.Network"
    server.save()

@task
def restart_queues():
    """
    tries to kill all celery workers and restarts them
    :return:
    """
    kill()
    local('fab startq:qextractor &')
    local('fab startq:qindexer &')
    local('fab startq:qretriever &')
    local('fab startq:qface &')
    local('fab startq:qfacedetector &')
    local('fab startq:qdetector &')

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
    from dvaapp.views import handle_uploaded_file, handle_youtube_video, pull_vdn_list\
        ,import_vdn_dataset_url
    from dvaapp.models import Video, Clusters,IndexEntries,TEvent,VDNServer
    from django.conf import settings
    from dvaapp.operations.query_processing import QueryProcessing
    from dvaapp.tasks import extract_frames, inception_index_by_id, perform_ssd_detection_by_id,\
        perform_yolo_detection_by_id, inception_index_regions_by_id, export_video_by_id, import_video_by_id,\
        execute_index_subquery, perform_clustering, assign_open_images_text_tags_by_id, perform_face_detection,\
        perform_face_indexing, segment_video
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name, False)
    for fname in glob.glob('tests/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name, False)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)
    # handle_youtube_video('world is not enough', 'https://www.youtube.com/watch?v=P-oNz3Nf50Q') # Temporarily disabled due error in travis
    for i,v in enumerate(Video.objects.all()):
        if v.dataset:
            extract_frames(TEvent.objects.create(video=v).pk)
        else:
            arguments_json =  json.dumps({'sync':True})
            segment_video(TEvent.objects.create(video=v,arguments_json=arguments_json).pk)
        inception_index_by_id(TEvent.objects.create(video=v).pk)
        if i ==0: # save travis time by just running detection on first video
            perform_ssd_detection_by_id(TEvent.objects.create(video=v).pk)
            perform_face_detection(TEvent.objects.create(video=v).pk)
            inception_index_regions_by_id(TEvent.objects.create(video=v).pk)
            assign_open_images_text_tags_by_id(TEvent.objects.create(video=v).pk)
        fname = export_video_by_id(TEvent.objects.create(video=v,event_type=TEvent.EXPORT).pk)
        f = SimpleUploadedFile(fname, file("{}/exports/{}".format(settings.MEDIA_ROOT,fname)).read(), content_type="application/zip")
        vimported = handle_uploaded_file(f, fname)
        import_video_by_id(TEvent.objects.create(video=vimported).pk)
    dc = Clusters()
    dc.indexer_algorithm = 'inception'
    dc.included_index_entries_pk = [k.pk for k in IndexEntries.objects.all().filter(algorithm=dc.indexer_algorithm)]
    dc.components = 32
    dc.save()
    clustering_task = TEvent()
    clustering_task.clustering = dc
    clustering_task.event_type = TEvent.CLUSTERING
    clustering_task.operation = 'perform_clustering'
    clustering_task.save()
    perform_clustering(clustering_task.pk)
    query_dict = {
        'image_data_b64':base64.encodestring(file('tests/query.png').read()),
        'indexers':[
            {
                'algorithm':'inception',
                'count':10,
                'approximate':False
            }
        ]
    }
    qp = QueryProcessing()
    qp.create_from_json(query_dict)
    execute_index_subquery(qp.indexer_queries[0].pk)
    query_dict = {
        'image_data_b64':base64.encodestring(file('tests/query.png').read()),
        'indexers':[
            {
                'algorithm':'inception',
                'count':10,
                'approximate':True
            }
        ]
    }
    qp = QueryProcessing()
    qp.create_from_json(query_dict)
    execute_index_subquery(qp.indexer_queries[0].pk)
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
    from dvaapp.tasks import perform_face_indexing
    for i,v in enumerate(Video.objects.all()):
        if i ==0: # save travis time by just running detection on first video
            perform_face_indexing(TEvent.objects.create(video=v).pk)


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
    :param detection: use fab launch_queues:1 to lauch detector queue in addition to all others
    :return:
    """
    local('fab startq:qextract &')
    local('fab startq:qindexer &')
    local('fab startq:qvgg &')
    local('fab startq:qretriever &')
    local('fab startq:qfaceretriever &')
    local('fab startq:qfacedetector &')
    local('fab startq:qclusterer &')
    local('fab startq:qdetector &')

@task
def launch_queues_env():
    """
    Launch workers for each queue
    :param detection: use fab launch_queues:1 to lauch detector queue in addition to all others
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video,VDNServer
    for k in os.environ:
        if k.startswith('LAUNCH_Q_'):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                local('fab startq:{},{} &'.format(queue_name,os.environ['LAUNCH_Q_qextract']))
            else:
                queue_name = k.split('_')[-1]
                local('fab startq:{} &'.format(queue_name))
    if not ('DISABLE_VDN' in os.environ):
        if VDNServer.objects.count() == 0:
            server = VDNServer()
            server.url = "http://www.visualdata.network/"
            server.name = "VisualData.Network"
            server.save()
    if 'TEST' in os.environ and Video.objects.count() == 0:
        test()


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
    from django.conf import settings
    mute = '--without-gossip --without-mingle --without-heartbeat' if 'CELERY_MUTE' in os.environ else ''
    if queue_name in settings.QUEUES:
        if queue_name == settings.Q_EXTRACTOR:
            command = 'celery -A dva worker -l info {} -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute,max(int(conc),2), queue_name,queue_name,queue_name)
            # TODO: worker fails due to
            # https://github.com/celery/celery/issues/3620
            # since the worker consumes from multiple queues
            #command = 'celery -A dva worker -l info {} -c {} -Q {},broadcast_tasks -n {}.%h -f logs/{}.log'.format(mute,int(conc), queue_name,queue_name,queue_name)
        else:
            command = 'celery -A dva worker -l info {} -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute,1, queue_name,queue_name,queue_name)
            # TODO: worker task fails same reason as above
            #command = 'celery -A dva worker -l info {} -P solo -c {} -Q {},broadcast_tasks -n {}.%h -f logs/{}.log'.format(mute,1, queue_name,queue_name,queue_name)
        logging.info(command)
        os.system(command)
    else:
        raise ValueError, "Queue {} not found".format(queue_name)



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
    from dvaapp.views import handle_uploaded_file, handle_youtube_video
    for fname in glob.glob('tests/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)
    handle_youtube_video('tomorrow never dies', 'https://www.youtube.com/watch?v=gYtz5sw98Bc')


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




@task
def process_video_list(filename):
    """
    submit multiple videos from a json file
    """
    import django,json
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.views import handle_youtube_video
    vlist = json.load(file(filename))
    for video in vlist:
        handle_youtube_video(video['name'],video['url'])


@task
def assign_tags(video_id):
    import django
    from PIL import Image
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp.models import Video,Frame,Region
    from dvalib import annotator
    from dvaapp.operations.video_processing import WVideo, WFrame
    dv = Video.objects.get(id=video_id)
    frames = Frame.objects.all().filter(video=dv)
    v = WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    wframes = {df.pk: WFrame(video=v, frame_index=df.frame_index, primary_key=df.pk) for df in frames}
    algorithm = annotator.OpenImagesAnnotator()
    logging.info("starting annotation {}".format(algorithm.name))
    for k,f in wframes.items():
        tags = algorithm.apply(f.local_path())
        a = Region()
        a.region_type = Region.ANNOTATION
        a.frame_id = k
        a.video_id = video_id
        a.object_name = "OpenImagesTag"
        a.metadata_text = " ".join([t for t,v in tags.iteritems() if v > 0.1])
        a.metadata_json = json.dumps({t:100.0*v for t,v in tags.iteritems() if v > 0.1})
        a.full_frame = True
        a.save()
        print a.metadata_text


def setup_django():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()




@task
def cluster():
    from lopq.utils import load_xvecs
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.tasks import perform_clustering
    from dvaapp.models import Video,Clusters,IndexEntries
    c = Clusters()
    c.indexer_algorithm = 'facenet'
    c.included_index_entries_pk = [k.pk for k in IndexEntries.objects.all() if k.algorithm == c.indexer_algorithm]
    c.components = 128
    c.cluster_count = 32
    c.save()
    perform_clustering(c.pk,True)

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
def heroku_reset(password):
    if raw_input("Are you sure type yes >>") == 'yes':
        local('heroku pg:reset DATABASE_URL')
        heroku_migrate()
        heroku_setup_vdn(password)
        local('heroku run python manage.py createsuperuser')


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
    from vdnapp.models import User,Dataset,Organization,Detector
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
        ('yolo_test_train_dataset_medium', 'https://www.dropbox.com/s/u1djt5obccczmcj/license_plates.zip'),
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
        ('Visual Genome objects ~16GB', 'us-east-1', 'visualdatanetwork', 'visual_genome.dva_export.zip'),
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
        d = Detector()
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
    from dvaapp.tasks import sync_bucket_video_by_id
    for v in Video.objects.all():
        e = TEvent()
        e.video_id = v.pk
        e.operation = 'sync_bucket_video_by_id'
        e.save()
        sync_bucket_video_by_id(e.pk)




@task
def detect_custom_objects(detector_pk,video_pk):
    """
    Detection using customized trained YOLO detectors
    :param detector_pk:
    :param video_pk:
    :return:
    """
    setup_django()
    from dvaapp.models import Region, Frame, CustomDetector
    from django.conf import settings
    from dvalib.yolo import trainer
    from PIL import Image
    args = {'detector_pk':int(detector_pk)}
    video_pk = int(video_pk)
    detector = CustomDetector.objects.get(pk=args['detector_pk'])
    args['root_dir'] = "{}/detectors/{}/".format(settings.MEDIA_ROOT, detector.pk)
    class_names = {k:v for k,v in json.loads(detector.class_names)}
    i_class_names = {i: k for k, i in class_names.items()}
    frames = {}
    for f in Frame.objects.all().filter(video_id=video_pk):
        frames[f.pk] = f
    images = []
    path_to_f = {}
    for k,f in frames.iteritems():
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,f.video_id,f.frame_index)
        path_to_f[path] = f
        images.append(path)
    train_task = trainer.YOLOTrainer(boxes=[], images=images, class_names=i_class_names, args=args,test_mode=True)
    results = train_task.predict()
    for path, box_class, score, top, left, bottom, right in results:
        r = Region()
        r.region_type = r.DETECTION
        r.confidence = int(100.0 * score)
        r.object_name = "YOLO_{}_{}".format(detector.pk, box_class)
        r.y = top
        r.x = left
        r.w = right - left
        r.h = bottom - top
        r.frame_id = path_to_f[path].pk
        r.video_id = path_to_f[path].video_id
        r.save()
        right = r.w + r.x
        bottom = r.h + r.y
        img = Image.open(path)
        img2 = img.crop((r.x,r.y,right, bottom))
        img2.save("{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT, video_pk, r.pk))


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
    from dvaapp.models import Region, Frame, CustomDetector, TEvent
    from dvaapp.shared import create_detector_folders, create_detector_dataset
    from dvalib.yolo import trainer
    start = TEvent.objects.get(pk=start_pk)
    args = json.loads(start.arguments_json)
    labels = set(args['labels']) if 'labels' in args else set()
    object_names = set(args['object_names']) if 'object_names' in args else set()
    detector = CustomDetector.objects.get(pk=args['detector_pk'])
    create_detector_folders(detector)
    args['root_dir'] = "{}/detectors/{}/".format(settings.MEDIA_ROOT,detector.pk)
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
    from dvaapp.shared import create_detector_folders
    import json
    from django.conf import settings
    from dvaapp.models import CustomDetector
    d = CustomDetector()
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
    create_detector_folders(d)
    shutil.copy("{}/phase_2_best.h5".format(path),"{}/detectors/{}/phase_2_best.h5".format(settings.MEDIA_ROOT,d.pk))


@task
def detect_text_boxes(video_pk,cpu_mode=False):
    """
    Detect Text Boxes in frames for a video using CTPN, must be run in dva_ctpn container
    :param detector_pk
    :param video_pk
    :return:
    """
    setup_django()
    from dvaapp.models import Region, Frame
    from django.conf import settings
    from PIL import Image
    import sys
    video_pk = int(video_pk)
    sys.path.append('/opt/ctpn/CTPN/tools/')
    sys.path.append('/opt/ctpn/CTPN/src/')
    from cfg import Config as cfg
    from other import resize_im, CaffeModel
    import cv2, caffe
    from detectors import TextProposalDetector, TextDetector
    NET_DEF_FILE = "/opt/ctpn/CTPN/models/deploy.prototxt"
    MODEL_FILE = "/opt/ctpn/CTPN/models/ctpn_trained_model.caffemodel"
    if cpu_mode:  # Set this to true for CPU only mode
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(cfg.TEST_GPU_ID)
    text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
    text_detector = TextDetector(text_proposals_detector)
    for f in Frame.objects.all().filter(video_id=video_pk):
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,video_pk,f.frame_index)
        im=cv2.imread(path)
        old_h,old_w, channels = im.shape
        im, _=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
        new_h, new_w, channels = im.shape
        mul_h = float(old_h)/float(new_h)
        mul_w = float(old_w)/float(new_w)
        text_lines=text_detector.detect(im)
        for k in text_lines:
            left, top, right, bottom ,score = k
            left, top, right, bottom = int(left*mul_w), int(top*mul_h), int(right*mul_w), int(bottom*mul_h)
            r = Region()
            r.region_type = r.DETECTION
            r.confidence = int(100.0 * score)
            r.object_name = "CTPN_TEXTBOX"
            r.y = top
            r.x = left
            r.w = right - left
            r.h = bottom - top
            r.frame_id = f.pk
            r.video_id = video_pk
            r.save()


@task
def recognize_text(video_pk):
    """
    Recognize text in regions with name CTPN_TEXTBOX using CRNN
    :param detector_pk
    :param video_pk
    :return:
    """
    setup_django()
    from dvaapp.models import Region
    from django.conf import settings
    from PIL import Image
    import sys
    video_pk = int(video_pk)
    import dvalib.crnn.utils as utils
    import dvalib.crnn.dataset as dataset
    import torch
    from torch.autograd import Variable
    from PIL import Image
    import dvalib.crnn.models.crnn as crnn
    model_path = '/root/DVA/dvalib/crnn/data/crnn.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = crnn.CRNN(32, 1, 37, 256, 1).cuda()
    model.load_state_dict(torch.load(model_path))
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))
    for r in Region.objects.all().filter(video_id=video_pk,object_name='CTPN_TEXTBOX'):
        img_path = "{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT,video_pk,r.pk)
        image = Image.open(img_path).convert('L')
        image = transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        dr = Region()
        dr.video_id = r.video_id
        dr.object_name = "CRNN_TEXT"
        dr.x = r.x
        dr.y = r.y
        dr.w = r.w
        dr.h = r.h
        dr.region_type = Region.ANNOTATION
        dr.metadata_text = sim_pred
        dr.frame_id = r.frame_id
        dr.save()


@task
def qt():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file
    from dvaapp.models import Video, TEvent
    from dvaapp.tasks import extract_frames,perform_face_detection,perform_face_indexing
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        v = handle_uploaded_file(f, name)
        # extract_frames(TEvent.objects.create(video=v).pk)
        # perform_face_detection(TEvent.objects.create(video=v).pk)
        # perform_face_indexing(TEvent.objects.create(video=v).pk)


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
def install_visual_data_notebook():
    """

    :return:
    """
    local('pip install --upgrade jupyter')
    local('pip install ipywidgets')
    local('jupyter nbextension enable --py --sys-prefix widgetsnbextension')


@task
def benchmark():
    with lcd('benchmarks/retrieval'):
        pass
