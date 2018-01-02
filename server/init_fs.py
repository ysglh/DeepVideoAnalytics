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
from dvaapp.models import TrainedModel, DVAPQL
from dvaapp.processing import DVAPQLProcess
from django.contrib.auth.models import User
from dvaapp.fs import get_path_to_file

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
    if ExternalServer.objects.count() == 0:
        for e in json.loads(file("../configs/custom_defaults/servers.json").read()):
            ExternalServer.objects.get_or_create(name=e['name'],url=e['url'])
    if sys.platform == 'darwin':
        default_models = json.loads(file("../configs/custom_defaults/trained_models_mac.json").read())
    else:
        default_models = json.loads(file("../configs/custom_defaults/trained_models.json").read())
    for m in default_models:
        if m['model_type'] == TrainedModel.DETECTOR:
            dm, created = TrainedModel.objects.get_or_create(name=m['name'],algorithm=m['algorithm'],mode=m['mode'],
                                                          files=m.get('files',[]), model_filename=m.get("filename", ""),
                                                          detector_type=m.get("detector_type", ""),
                                                          arguments=m.get("arguments", {}),
                                                          model_type=TrainedModel.DETECTOR,)
            if created:
                dm.download()
        else:
            dm, created = TrainedModel.objects.get_or_create(name=m['name'], mode=m.get('mode',TrainedModel.TENSORFLOW),
                                                             files=m.get('files',[]),
                                                             algorithm=m.get('algorithm',""),
                                                             arguments=m.get("arguments", {}),
                                                             shasum=m.get('shasum',None),
                                                             model_type=m['model_type'])
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
