#!/usr/bin/env python
import django
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.models import TEvent
from dvaapp.task_handlers import handle_perform_analysis, handle_perform_indexing, handle_perform_detection


if '__name__' == '__main__':
    task_name = sys.argv[-2]
    pk = int(sys.argv[-1])
    if task_name == 'perform_indexing':
        handle_perform_indexing(TEvent.objects.get(pk=pk))
    elif task_name == 'perform_detection':
        handle_perform_detection(TEvent.objects.get(pk=pk))
    elif task_name == 'perform_analysis':
        handle_perform_analysis(TEvent.objects.get(pk=pk))
    else:
        raise ValueError("Unknown task name {}".format(task_name))
