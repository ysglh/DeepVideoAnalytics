import os,logging,time,boto3, glob,subprocess,calendar,sys
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
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
def start_server_container(perform_test=False):
    """
    Start sever container WITHOUT using nginx and uwsgi
    :param test:
    :return:
    """
    local('sleep 60')
    migrate()
    launch_queues(True)
    if perform_test:
        test()
    local('python manage.py runserver 0.0.0.0:8000')

@task
def setup_container_gpu():
    local('sleep 60')
    migrate()
    local('chmod 0777 -R /tmp')
    try:
        local("mv docker_GPU/configs/nginx.conf /etc/nginx/")
    except:
        print "warning assuming that the config was already moved"
        pass
    try:
        local("mv docker_GPU/configs/nginx-app.conf /etc/nginx/sites-available/default")
    except:
        print "warning assuming that the config was already moved"
        pass
    try:
        local("mv docker_GPU/configs/supervisor-app.conf /etc/supervisor/conf.d/")
    except:
        print "warning assuming that the config was already moved"
        pass
    local("python manage.py collectstatic --no-input")
    local("chmod 0777 -R dva/staticfiles/")
    local("chmod 0777 -R dva/media/")


@task
def start_server_container_gpu(perform_test=False):
    """
    Start sever container using nginx and uwsgi
    :param test:
    :return:
    """
    setup_container_gpu()
    launch_queues(True)
    if perform_test:
        test()
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
    if sys.platform == 'darwin':
        for qname in set(settings.TASK_NAMES_TO_QUEUE.values()):
            try:
                local('rabbitmqadmin purge queue name={}'.format(qname))
            except:
                logging.warning("coudnt clear queue {}".format(qname))
    local('python manage.py makemigrations')
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


@task
def restart_queues(detection=False):
    """
    tries to kill all celery workers and restarts them
    :return:
    """
    kill_queues()
    local('fab startq:extractor &')
    local('fab startq:indexer &')
    local('fab startq:retriever &')
    local('fab startq:face &')
    local('fab startq:facedetector &')
    if detection:
        local('fab startq:detector &')

@task
def kill_queues():
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
    from dvaapp.tasks import extract_frames, perform_face_indexing, inpcetion_index_by_id, perform_yolo_ssd_detection_by_id
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
        inpcetion_index_by_id(v.pk)
        if i ==0: # save travis time by just running detection on first video
            perform_yolo_ssd_detection_by_id(v.pk)
            perform_face_indexing(v.pk)
    test_backup()


@task
def quick_test(detection=False):
    """
    Used on my local Mac for quickly cleaning and testing
    :param detection:
    :return:
    """
    clean()
    create_super()
    test()
    launch_queues(detection)


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
def create_super():
    """
    Create a superuser
    :return:
    """
    local('echo "from django.contrib.auth.models import User; User.objects.create_superuser(\'akshay\', \'akshay@test.com\', \'super\')" | python manage.py shell')


@task
def launch_queues(detection=False):
    """
    Launch workers for each queue
    :param detection: use fab launch_queues:1 to lauch detector queue in addition to all others
    :return:
    """
    local('fab startq:extractor &')
    local('fab startq:indexer &')
    local('fab startq:retriever &')
    local('fab startq:face &')
    local('fab startq:facedetector &')
    if detection:
        local('fab startq:detector &')


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
    Q_INDEXER = settings.Q_INDEXER
    Q_EXTRACTOR = settings.Q_EXTRACTOR
    Q_DETECTOR = settings.Q_DETECTOR
    Q_RETRIEVER = settings.Q_RETRIEVER
    Q_FACE = settings.Q_FACE_RETRIEVER
    Q_FACEDETECTOR = settings.Q_FACE_DETECTOR
    if queue_name == 'indexer':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_INDEXER, Q_INDEXER,Q_INDEXER)
    elif queue_name == 'extractor':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_EXTRACTOR,Q_EXTRACTOR,Q_EXTRACTOR)
    elif queue_name == 'detector':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_DETECTOR,Q_DETECTOR, Q_DETECTOR)
    elif queue_name == 'retriever':
        command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_RETRIEVER,Q_RETRIEVER,Q_RETRIEVER)
    elif queue_name == 'face':
        command = 'celery -A dva worker -l info -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_FACE,Q_FACE,Q_FACE)
    elif queue_name == 'facedetector':
        command = 'celery -A dva worker -l info -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_FACEDETECTOR,Q_FACEDETECTOR,Q_FACEDETECTOR)
    else:
        raise NotImplementedError
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
def detect(video_id):
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
    detector_list = {'ssd': detector.SSDetector(),}
    if 'YOLO_ENABLE' in os.environ:
        detector_list['yolo'] = detector.YOLODetector()
    for alogrithm in detector_list.itervalues():
        logging.info("starting detection {}".format(alogrithm.name))
        frame_detections = alogrithm.detect(wframes.values())
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
    dv.detections = detection_count
    dv.save()

@task
def pyscenedetect(video_id,rescaled_width=600):
    """
    Pyscenedetect often causes unexplainable double free errors on some machine when executing cap.release()
    This ensures that the task recovers even if the command running inside a subprocess fails
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
    manager = pyscenecustom.manager.SceneManager(save_image_prefix="{}/{}/frames/".format(settings.MEDIA_ROOT, video_id), rescaled_width=int(rescaled_width))
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
    products.prepare(input_dir)
    products.build_approximate()

@task
def push_external_products_index(path='/Users/aub3/temptest/products/'):
    sys.path.append(os.path.dirname(__file__))
    from dvalib import external_indexed
    products = external_indexed.ProductsIndex(path=path)
    products.push_to_s3()
