import django
import sys, os

sys.path.append(os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.tasks import perform_detection, perform_indexing, perform_analysis, perform_segmentation


if '__name__' == '__main__':
    task_name = sys.argv[-2]
    pk = int(sys.argv[-1])
    if task_name == 'perform_indexing':
        perform_indexing(pk)
    elif task_name == 'perform_detection':
        perform_detection(pk)
    elif task_name == 'perform_analysis':
        perform_analysis(pk)
    elif task_name == 'perform_segmentation':
        perform_segmentation(pk)