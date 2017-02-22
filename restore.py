import sys,dva,os
import subprocess
import django,os,glob


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    media_dir = settings.MEDIA_ROOT
    current_path = os.path.abspath(os.path.dirname(__file__))
    command = ['unzip','-o','{}/backups/backup.zip'.format(current_path)]
    print ' '.join(command)
    zipper = subprocess.Popen(command,cwd=media_dir)
    zipper.communicate()
    db = settings.DATABASES.values()[0]
    pg = '/Users/aub3/PostgreSQL/pg96/bin/psql' if sys.platform == 'darwin' else 'psql'
    with open('{}/postgres.dump'.format(media_dir)) as dumpfile:
        dump = subprocess.Popen([pg,'--dbname','postgresql://{}:{}@{}:5432/{}'.format(db['USER'],db['PASSWORD'],db['HOST'],db['NAME'])],cwd=media_dir,stdin=dumpfile)
        dump.communicate()
    print dump.returncode
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode