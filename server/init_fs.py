#!/usr/bin/env python
import django, json, sys, os, logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/init_fs.log',
                    filemode='a')
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
from dvaui.models import ExternalServer
from dvaapp.models import TrainedModel, Retriever, DVAPQL
from dvaui.defaults import EXTERNAL_SERVERS
from dvaapp.processing import DVAPQLProcess
from django.contrib.auth.models import User
from dvaapp.fs import get_path_to_file
from django.utils import timezone
from dvaui.defaults import DEFAULT_MODELS

if __name__ == "__main__":
    if not User.objects.filter(is_superuser=True).exists() and 'SUPERUSER' in os.environ:
        User.objects.create_superuser(username=os.environ['SUPERUSER'], password=os.environ['SUPERPASS'],
                                      email=os.environ['SUPEREMAIL'])
    for create_dirname in ['queries', 'exports', 'external', 'retrievers', 'ingest','training_sets']:
        if not os.path.isdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname)):
            try:
                os.mkdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname))
            except:
                pass
    for e in EXTERNAL_SERVERS:
        ExternalServer.objects.get_or_create(name=e['name'],url=e['url'])
    for m in DEFAULT_MODELS:
        if m['model_type'] == "detector":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'],algorithm=m['algorithm'],mode=m['mode'],
                                                          files=m.get('files',[]), model_filename=m.get("filename", ""),
                                                          detector_type=m.get("detector_type", ""),
                                                          class_index_to_string=m.get("class_index_to_string", {}),
                                                          model_type=TrainedModel.DETECTOR,)
            if created:
                dm.download()
        if m['model_type'] == "indexer":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'], mode=m['mode'], files=m.get('files',[]),
                                                          shasum=m['shasum'],model_type=TrainedModel.INDEXER)
            if created:
                dr, dcreated = Retriever.objects.get_or_create(name=m['name'],
                                                               source_filters={'indexer_shasum': dm.shasum})
                if dcreated:
                    dr.last_built = timezone.now()
                    dr.save()
            if created:
                dm.download()
        if m['model_type'] == "analyzer":
            dm, created = TrainedModel.objects.get_or_create(name=m['name'], files=m.get('files',[]), mode=m['mode'],
                                                          model_type=TrainedModel.ANALYZER)
            if created:
                dm.download()
    if 'INIT_PROCESS' in os.environ and DVAPQL.objects.count() == 0:
        path = os.environ.get('INIT_PROCESS')
        p = DVAPQLProcess()
        if not path.startswith('/root/DVA/configs/custom_defaults/'):
            get_path_to_file(path,"temp.json")
            path = 'temp.json'
        try:
            jspec = json.load(file(path))
        except:
            logging.exception("could not load : {}".format(path))
        else:
            p.create_from_json(jspec)
            p.launch()
