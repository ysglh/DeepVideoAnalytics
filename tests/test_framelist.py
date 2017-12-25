import django, sys, glob, os

sys.path.append(os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaui.view_shared import handle_uploaded_file
from dvaapp.tasks import perform_import, perform_frame_download
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaapp.models import TEvent

if __name__ == '__main__':
    for fname in glob.glob('ci/framelist.*'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/json")
        v = handle_uploaded_file(f, name)
        dt = TEvent.objects.get(video=v, operation='perform_import')
        perform_import(dt.pk)
        for t in TEvent.objects.filter(parent=dt):
            perform_frame_download(t.pk)
