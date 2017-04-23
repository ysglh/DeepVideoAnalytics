import os,logging,time,boto3, glob,subprocess,calendar,sys
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
import json
import random
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
        migrate()
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
        local("chmod 0777 -R dva/media/")
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
def restart_queues(detection=False):
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
    if detection:
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
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file, handle_youtube_video
    from dvaapp.models import Video
    from django.conf import settings
    from dvaapp.tasks import extract_frames, perform_face_indexing, inception_index_by_id, perform_ssd_detection_by_id, perform_yolo_detection_by_id, inception_index_ssd_detection_by_id, export_video_by_id, import_video_by_id
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name, False)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)
    handle_youtube_video('tomorrow never dies', 'https://www.youtube.com/watch?v=gYtz5sw98Bc')
    for i,v in enumerate(Video.objects.all()):
        extract_frames(v.pk)
        inception_index_by_id(v.pk)
        if i ==0: # save travis time by just running detection on first video
            perform_ssd_detection_by_id(v.pk)
            perform_yolo_detection_by_id(v.pk)
            perform_face_indexing(v.pk)
            inception_index_ssd_detection_by_id(v.pk)
        fname = export_video_by_id(v.pk)
        f = SimpleUploadedFile(fname, file("{}/exports/{}".format(settings.MEDIA_ROOT,fname)).read(), content_type="application/zip")
        vimported = handle_uploaded_file(f, fname)
        import_video_by_id(vimported.pk)
    test_backup()


@task
def quick(detection=False):
    """
    Used on my local Mac for quickly cleaning and testing
    :param detection:
    :return:
    """
    clean()
    superu()
    test()
    launch(detection)


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
def launch(detection=False):
    """
    Launch workers for each queue
    :param detection: use fab launch_queues:1 to lauch detector queue in addition to all others
    :return:
    """
    local('fab startq:qextract &')
    local('fab startq:qindexer &')
    local('fab startq:qretriever &')
    local('fab startq:qfaceretriever &')
    local('fab startq:qfacedetector &')
    local('fab startq:qclusterer &')
    if detection:
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
def startq(queue_name):
    """
    Start worker to handle a queue, Usage: fab startq:indexer
    Concurrency is set to 1 but you can edit code to change.
    :param queue_name: indexer, extractor, retriever, detector
    :return:
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    if queue_name in settings.QUEUES:
        if queue_name == settings.Q_EXTRACTOR:
            command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, queue_name,queue_name,queue_name)
        else:
            command = 'celery -A dva worker -l info -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, queue_name,queue_name,queue_name)
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
def yolo_detect(video_id):
    """
    This is a HACK since Tensorflow is absolutely atrocious in allocating and freeing up memory.
    Once a process / session is allocated a memory it cannot be forced to clear it up.
    As a result this code gets called via a subprocess which clears memory when it exits.

    :param video_id:
    :return:
    """
    import django
    from PIL import Image
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp.models import Video,Detection,Frame
    from dvalib import entity,detector
    dv = Video.objects.get(id=video_id)
    frames = Frame.objects.all().filter(video=dv)
    v = entity.WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    wframes = {df.pk: entity.WFrame(video=v, frame_index=df.frame_index, primary_key=df.pk) for df in frames}
    detection_count = 0
    algorithm = detector.YOLODetector()
    logging.info("starting detection {}".format(algorithm.name))
    frame_detections = algorithm.detect(wframes.values())
    for frame_pk,detections in frame_detections.iteritems():
        for d in detections:
            dd = Detection()
            dd.video = dv
            dd.frame_id = frame_pk
            dd.object_name = d['name']
            dd.confidence = d['confidence']
            dd.x = d['left']
            dd.y = d['top']
            dd.w = d['right'] - d['left']
            dd.h = d['bot'] - d['top']
            dd.save()
            img = Image.open(wframes[frame_pk].local_path())
            img2 = img.crop((d['left'], d['top'], d['right'], d['bot']))
            img2.save("{}/{}/detections/{}.jpg".format(settings.MEDIA_ROOT, video_id, dd.pk))
            detection_count += 1
    dv.refresh_from_db()
    dv.detections = dv.detections + detection_count
    dv.save()


@task
def ssd_detect(video_id):
    """
    This is a HACK since Tensorflow is absolutely atrocious in allocating and freeing up memory.
    Once a process / session is allocated a memory it cannot be forced to clear it up.
    As a result this code gets called via a subprocess which clears memory when it exits.

    :param video_id:
    :return:
    """
    import django
    from PIL import Image
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp.models import Video,Detection,Frame
    from dvalib import entity,detector
    dv = Video.objects.get(id=video_id)
    frames = Frame.objects.all().filter(video=dv)
    v = entity.WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    wframes = {df.pk: entity.WFrame(video=v, frame_index=df.frame_index, primary_key=df.pk) for df in frames}
    detection_count = 0
    algorithm = detector.SSDetector()
    logging.info("starting detection {}".format(algorithm.name))
    frame_detections = algorithm.detect(wframes.values())
    for frame_pk,detections in frame_detections.iteritems():
        for d in detections:
            dd = Detection()
            dd.video = dv
            dd.frame_id = frame_pk
            dd.object_name = d['name']
            dd.confidence = d['confidence']
            dd.x = d['left']
            dd.y = d['top']
            dd.w = d['right'] - d['left']
            dd.h = d['bot'] - d['top']
            dd.save()
            img = Image.open(wframes[frame_pk].local_path())
            img2 = img.crop((d['left'], d['top'], d['right'], d['bot']))
            img2.save("{}/{}/detections/{}.jpg".format(settings.MEDIA_ROOT, video_id, dd.pk))
            detection_count += 1
    dv.refresh_from_db()
    dv.detections = dv.detections + detection_count
    dv.save()


@task
def pyscenedetect(video_id,rescaled_width=600,rescale=True):
    """
    Pyscenedetect often causes unexplainable double free errors on some machine when executing cap.release()
    This ensures that the task recovers partial frame data i.e. every nth frame even if the command running inside a subprocess fails
    :param video_id:
    :param rescaled_width:
    :return:
    """
    import django
    from PIL import Image
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video
    from django.conf import settings
    from dvalib import pyscenecustom,entity
    dv = Video.objects.get(id=video_id)
    v = entity.WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    if 'RESCALE_DISABLE' in os.environ:
        rescale = False
    manager = pyscenecustom.manager.SceneManager(save_image_prefix="{}/{}/frames/".format(settings.MEDIA_ROOT, video_id), rescaled_width=int(rescaled_width),rescale=rescale)
    pyscenecustom.detect_scenes_file(v.local_path, manager)


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
def perform_face_detection(video_id):
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.tasks import perform_face_indexing
    perform_face_indexing(int(video_id))


def setup_django():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()


@task
def build_external_products_index(input_dir='/Users/aub3/temptest/gtin/', output_dir="/Users/aub3/temptest/products"):
    """
    Build external index for products
    :param input_dir:
    :param output_dir:
    :return:
    """
    sys.path.append(os.path.dirname(__file__))
    from dvalib import external_indexed
    products = external_indexed.ProductsIndex(path=output_dir)
    # products.prepare(input_dir)
    products.build_approximate()


@task
def push_external_products_index(path='/Users/aub3/temptest/products/'):
    sys.path.append(os.path.dirname(__file__))
    from dvalib import external_indexed
    products = external_indexed.ProductsIndex(path=path)
    products.push_to_s3()


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
    local('heroku run python vdn_manage.py migrate')


@task
def heroku_update_env():
    local('heroku config:get DATABASE_URL > db.env')


@task
def heroku_shell():
    local('heroku run python vdn_manage.py shell')


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
def heroku_make_migrate():
    local('python vdn_manage.py makemigrations')


@task
def heroku_dbflush():
    local('heroku pg:reset DATABASE_URL')
    heroku_migrate()
    local('heroku run python vdn_manage.py createsuperuser')

@task
def heroku_local_static():
    local('python vdn_manage.py collectstatic')


@task
def heroku_migrate():
    local('heroku run python vdn_manage.py migrate')

def get_coco_dirname():
    if sys.platform == 'darwin':
        dirname = '/Users/aub3/coco_input/'
    else:
        dirname = 'coco_input'
    return dirname


@task
def download_coco(size=500):
    dirname = get_coco_dirname()
    try:
        os.mkdir(dirname)
        with lcd(dirname):
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip")
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip")
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip")
            local('unzip "*.zip"')
    except:
        pass
    train_data = json.load(file("{}/annotations/instances_train2014.json".format(dirname)))
    captions_train_data = json.load(file("{}/annotations/captions_train2014.json".format(dirname)))
    keypoints_train_data = json.load(file("{}/annotations/person_keypoints_train2014.json".format(dirname)))
    sample = random.sample(train_data['images'], int(size))
    ids = set()
    for count, img in enumerate(sample):
        if (count + 1) % 100 == 0:
            print count
        fname = os.path.join(dirname, img['file_name'])
        if not os.path.exists(fname):
            urlretrieve(img['coco_url'], fname)
        ids.add(img['id'])
    data = defaultdict(lambda: {'image': None, 'annotations': [], 'captions': [], 'keypoints': []})
    id_to_license = {k['id']: k for k in train_data['licenses']}
    id_to_category = {k['id']: k for k in train_data['categories']}
    kp_id_to_category = {k['id']: k for k in keypoints_train_data['categories']}
    for entry in train_data['images']:
        if entry['id'] in ids:
            entry['license'] = id_to_license[entry['license']]
            data[entry['id']]['image'] = entry
    for annotation in train_data['annotations']:
        if annotation['image_id'] in ids:
            annotation['category'] = id_to_category[annotation['category_id']]
            data[annotation['image_id']]['annotations'].append(annotation)
    del train_data
    for annotation in captions_train_data['annotations']:
        if annotation['image_id'] in ids:
            data[annotation['image_id']]['captions'].append(annotation)
    del captions_train_data
    for annotation in keypoints_train_data['annotations']:
        if annotation['image_id'] in ids:
            annotation['category'] = kp_id_to_category[annotation['category_id']]
            data[annotation['image_id']]['keypoints'].append(annotation)
    del keypoints_train_data
    with open('{}/coco_sample_metadata.json'.format(dirname), 'w') as output:
        json.dump(data, output)


@task
def generate_vdn(fast=False):
    kill()
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.views import handle_uploaded_file, handle_youtube_video
    from dvaapp import models
    from dvaapp.tasks import extract_frames, perform_face_detection_indexing_by_id, inception_index_by_id, \
        perform_ssd_detection_by_id, perform_yolo_detection_by_id, inception_index_ssd_detection_by_id, \
        export_video_by_id
    dirname = get_coco_dirname()
    local('wget https://www.dropbox.com/s/2dq085iu34y0hdv/coco_input.zip?dl=1 -O coco.zip')
    local('unzip coco.zip')
    with lcd(dirname):
        local("zip coco_input.zip -r *.jpg")
    fname = '{}/coco_input.zip'.format(dirname)
    with open('{}/coco_sample_metadata.json'.format(dirname)) as datafile:
        data = json.load(datafile)
    f = SimpleUploadedFile("coco_input.zip", file(fname).read(), content_type="application/zip")
    v = handle_uploaded_file(f, 'mscoco_sample_500')
    extract_frames(v.pk)
    video = v
    models.Annotation.objects.all().filter(video=video).delete()
    for frame in models.Frame.objects.all().filter(video=video):
        frame_id = str(int(frame.name.split('_')[-1].split('.')[0]))
        annotation = models.Annotation()
        annotation.video = v
        annotation.frame = frame
        annotation.full_frame = True
        annotation.metadata_json = json.dumps(data[frame_id]['image'])
        annotation.label = 'metadata'
        annotation.save()
    for frame in models.Frame.objects.all().filter(video=video):
        frame_id = str(int(frame.name.split('_')[-1].split('.')[0]))
        for a in data[frame_id][u'annotations']:
            annotation = models.Annotation()
            annotation.video = v
            annotation.frame = frame
            annotation.metadata_json = json.dumps(a)
            annotation.full_frame = False
            annotation.x = a['bbox'][0]
            annotation.y = a['bbox'][1]
            annotation.w = a['bbox'][2]
            annotation.h = a['bbox'][3]
            label, _ = models.VLabel.objects.get_or_create(video=video, label_name='coco_instance/{}/{}'.format(
                a[u'category'][u'supercategory'], a[u'category'][u'name']))
            annotation.label = label.label_name
            annotation.label_parent = label
            annotation.save()
        for a in data[frame_id][u'keypoints']:
            annotation = models.Annotation()
            annotation.video = v
            annotation.frame = frame
            annotation.metadata_json = json.dumps(a)
            annotation.x = a['bbox'][0]
            annotation.y = a['bbox'][1]
            annotation.w = a['bbox'][2]
            annotation.h = a['bbox'][3]
            label, _ = models.VLabel.objects.get_or_create(video=video, label_name='coco_keypoints/{}/{}'.format(
                a[u'category'][u'supercategory'], a[u'category'][u'name']))
            annotation.label = label.label_name
            annotation.label_parent = label
            annotation.save()
        for caption in data[frame_id][u'captions']:
            annotation = models.Annotation()
            annotation.video = v
            annotation.frame = frame
            annotation.metadata_text = caption['caption']
            annotation.full_frame = True
            label, _ = models.VLabel.objects.get_or_create(video=video, label_name='coco_caption')
            annotation.label = label.label_name
            annotation.label_parent = label
            annotation.save()
    if not fast:
        inception_index_by_id(v.pk)
        perform_ssd_detection_by_id(v.pk)
        perform_face_detection_indexing_by_id(v.pk)
        inception_index_ssd_detection_by_id(v.pk)
    export_video_by_id(v.pk)
    v = handle_youtube_video("Zelda","https://www.youtube.com/watch?v=vHiTxNrbB4M")
    extract_frames(v.pk)
    if not fast:
        inception_index_by_id(v.pk)
        perform_ssd_detection_by_id(v.pk)
        perform_face_detection_indexing_by_id(v.pk)
        inception_index_ssd_detection_by_id(v.pk)
    export_video_by_id(v.pk)
    v = handle_youtube_video("Paris","https://www.youtube.com/watch?v=zEAqJmS6ajk")
    extract_frames(v.pk)
    if not fast:
        inception_index_by_id(v.pk)
        perform_ssd_detection_by_id(v.pk)
        perform_face_detection_indexing_by_id(v.pk)
        inception_index_ssd_detection_by_id(v.pk)
    export_video_by_id(v.pk)
    local('wget https://www.dropbox.com/s/g8dv5yeh9bmflec/lfw_funneled.zip?dl=1 -O lfw.zip')
    f = SimpleUploadedFile("lfw.zip", file("lfw.zip").read(), content_type="application/zip")
    v = handle_uploaded_file(f, 'LFW subset')
    extract_frames(v.pk)
    if not fast:
        inception_index_by_id(v.pk)
        perform_face_detection_indexing_by_id(v.pk)
    export_video_by_id(v.pk)