import sys,dva,os
import subprocess
import django,os,glob
import time,calendar


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    media_dir = settings.MEDIA_ROOT
    if len(sys.argv) == 1:
        print "usage 'python backup_restory.py backup' or 'python backup_restore.py restore /path/to/backup_12354668.zip'"
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
        current_path = os.path.abspath(os.path.dirname(__file__))
        command = ['unzip', '-o', sys.argv[2]]
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
        print "usage 'python backup_restory.py backup' or 'python backup_restore.py restore /path/to/backup_12354668.zip'"