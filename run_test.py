import django,os,glob
from dvalib import indexer
from django.core.files.uploadedfile import SimpleUploadedFile


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp import tasks
    from dvaapp.views import handle_uploaded_file
    from dvaapp.models import Video
    for fname in glob.glob('tests/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f,name)
