import django
import sys, os, glob
root_path = os.path.join(os.path.dirname(__file__),'../')
print root_path
sys.path.append(root_path)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaui.view_shared import handle_uploaded_file
from dvaapp.tasks import perform_import
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaapp.models import TEvent
for fname in glob.glob('../framelist.*'):
    name = fname.split('/')[-1].split('.')[0]
    f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/json")
    v = handle_uploaded_file(f, name)
    perform_import(TEvent.objects.get(video=v, operation='perform_import').pk)
