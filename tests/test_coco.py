import django, os, sys, glob, shutil
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaui.view_shared import handle_uploaded_file
from dvaapp.tasks import perform_import, perform_region_import, perform_dataset_extraction
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaapp.models import TEvent
from django.conf import settings

if __name__ == '__main__':

    for fname in glob.glob('ci/coco*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        v = handle_uploaded_file(f, name)
        dt = TEvent.objects.get(video=v, operation='perform_import')
        perform_import(dt.pk)
        dt = TEvent(video=v, operation='perform_dataset_extraction')
        dt.save()
        perform_dataset_extraction(dt.pk)
        shutil.copy("ci/coco_regions/coco_ci_regions.json", "{}/ingest/coco_ci_regions.json".format(settings.MEDIA_ROOT))
        args = {"path": "/ingest/coco_ci_regions.json"}
        dt = TEvent(video=v, operation='perform_region_import', arguments=args)
        dt.save()
        perform_region_import(dt.pk)