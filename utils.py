import sys,dva,os
import subprocess
import django,os,glob
import time,calendar
from django.core.files.uploadedfile import SimpleUploadedFile

USAGE = """
'python utils.py test'
'python utils.py ci_test'
'python utils.py startq indexer'
'python utils.py startq extractor 3'
'python utils.py startq detector'
'python utils.py startq retriever'
'python utils.py backup'
'python utils.py restore /path/to/backup_12354668.zip'
"""

if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp.views import handle_uploaded_file, handle_youtube_video
    from dvaapp.models import Video
    from dvaapp.tasks import extract_frames, perform_indexing, perform_detection
    media_dir = settings.MEDIA_ROOT
    Q_INDEXER = settings.Q_INDEXER
    Q_EXTRACTOR = settings.Q_EXTRACTOR
    Q_DETECTOR = settings.Q_DETECTOR
    Q_RETRIEVER = settings.Q_RETRIEVER
    if len(sys.argv) == 1:
        print USAGE
    elif sys.argv[1] == 'startq':
        # Tasks running on GPU should have concurrency set to 1 otherwise, it might prosent an issue with GPU Memory allocation
        if sys.argv[2] == 'indexer':
            command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_INDEXER, Q_INDEXER,Q_INDEXER)
        elif sys.argv[2] == 'extractor':
            if len(sys.argv) > 3:
                concurrency = int(sys.argv[3])
            else:
                concurrency = 1
            command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(concurrency,Q_EXTRACTOR,Q_EXTRACTOR,Q_EXTRACTOR)
        elif sys.argv[2] == 'detector':
            command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_DETECTOR,Q_DETECTOR, Q_DETECTOR)
        elif sys.argv[2] == 'retriever':
            command = 'celery -A dva worker -l info -c {} -Q {} -n {}.%h -f logs/{}.log'.format(1, Q_RETRIEVER,Q_RETRIEVER,Q_RETRIEVER)
        else:
            raise NotImplementedError
        print command
        os.system(command)
    elif sys.argv[1] == 'test':
        for fname in glob.glob('tests/*.mp4'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
            handle_uploaded_file(f, name)
        for fname in glob.glob('tests/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
        handle_youtube_video('jungle book', 'https://www.youtube.com/watch?v=C4qgAaxB_pc')
    elif sys.argv[1] == 'ci_test':
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
    elif sys.argv[1] == 'backup':
        try:
            os.mkdir('backups')
        except:
            pass
        media_dir = settings.MEDIA_ROOT
        db = settings.DATABASES.values()[0]
        pg = '/Users/aub3/PostgreSQL/pg96/bin/pg_dump' if sys.platform == 'darwin' else 'pg_dump'
        with open('{}/postgres.dump'.format(media_dir),'w') as dumpfile:
            dump = subprocess.Popen([pg,'--clean','--dbname','postgresql://{}:{}@{}:5432/{}'.format(db['USER'],db['PASSWORD'],db['HOST'],db['NAME'])],cwd=media_dir,stdout=dumpfile)
            dump.communicate()
        print dump.returncode
        current_path = os.path.abspath(os.path.dirname(__file__))
        command = ['zip','-r','{}/backups/backup_{}.zip'.format(current_path,calendar.timegm(time.gmtime())),'.']
        print ' '.join(command)
        zipper = subprocess.Popen(command,cwd=media_dir)
        zipper.communicate()
        os.remove('{}/postgres.dump'.format(media_dir))
        print zipper.returncode
    elif sys.argv[1] == 'restore':
        if len(sys.argv) == 2:
            print USAGE
        current_path = os.path.abspath(os.path.dirname(__file__))
        command = ['unzip', '-o', '{}'.format(os.path.join(current_path,sys.argv[2]))]
        print ' '.join(command)
        zipper = subprocess.Popen(command, cwd=media_dir)
        zipper.communicate()
        db = settings.DATABASES.values()[0]
        pg = '/Users/aub3/PostgreSQL/pg96/bin/psql' if sys.platform == 'darwin' else 'psql'
        with open('{}/postgres.dump'.format(media_dir)) as dumpfile:
            dump = subprocess.Popen([pg, '--dbname','postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'],db['NAME'])], cwd=media_dir, stdin=dumpfile)
            dump.communicate()
        print dump.returncode
        os.remove('{}/postgres.dump'.format(media_dir))
        print zipper.returncode
    else:
        print USAGE