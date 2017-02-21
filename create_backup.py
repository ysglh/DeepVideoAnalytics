import sys,dva,os
import subprocess

import django,os,glob
from django.core.files.uploadedfile import SimpleUploadedFile


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    media_dir = settings.MEDIA_ROOT
    db = settings.DATABASES.values()[0]
    dump = subprocess.Popen(['pg_dump','--dbname','postgresql://{}:{}@{}:5432/dvadb'.format(db['USER'],db['PASSWORD'],db['HOST'])],cwd=media_dir)
    dump.communicate()