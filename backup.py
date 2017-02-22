import sys,dva,os
import subprocess
import django,os,glob


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    try:
        os.remove('backups/backup.zip')
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
    command = ['zip','-r','{}/backups/backup.zip'.format(current_path),'.']
    print ' '.join(command)
    zipper = subprocess.Popen(command,cwd=media_dir)
    zipper.communicate()
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode