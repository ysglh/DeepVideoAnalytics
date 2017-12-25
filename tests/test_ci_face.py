#!/usr/bin/env python
import django, sys, os
sys.path.append('../server/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.models import Video, TEvent
from dvaapp.tasks import perform_indexing

if __name__ == '__main__':
    for i, v in enumerate(Video.objects.all()):
        if i == 0:  # save travis time by just running detection on first video
            args = {
                'filter': {'object_name__startswith': 'MTCNN_face'},
                'index': 'facenet',
                'target': 'regions'}
            perform_indexing(TEvent.objects.create(video=v, arguments=args).pk)