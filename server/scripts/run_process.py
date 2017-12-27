#!/usr/bin/env python
import django, sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.processing import DVAPQLProcess

if __name__ == '__main__':
    path = sys.argv[-1]
    with open(path) as f:
        j = json.load(f)
    p = DVAPQLProcess()
    p.create_from_json(j)
    p.launch()
    print "launched Process with id {} ".format(p.process.pk)
