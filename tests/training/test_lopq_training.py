import os, sys, shutil, json
import django
sys.path.append("../../server/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
from dvaapp.models import TrainedModel, Retriever
from dvalib.trainers import lopq_trainer
import numpy as np


if __name__ == '__main__':
    l = lopq_trainer.LOPQTrainer(name="Facenet_LOPQ_on_LFW",
                                 dirname=os.path.join(os.path.dirname('__file__'),"../../shared/facenet_lopq/"),
                                 components=64,m=32,v=32,sub=256,
                                 source_indexer_shashum="9f99caccbc75dcee8cb0a55a0551d7c5cb8a6836")
    data = np.load('facenet.npy')
    l.train(data)
    j = l.save()
    with open("lopq_facenet_approximator.json",'w') as out:
        json.dump(j,out)
    m = TrainedModel(**j)
    m.save()
    m.create_directory()
    for f in m.files:
        shutil.copy(f['url'],'{}/models/{}/{}'.format(settings.MEDIA_ROOT,m.pk,f['filename']))
    dr = Retriever.objects.create(name="lopq retriever",source_filters={},
                                  algorithm=Retriever.LOPQ,
                                  approximator_shashum=m.shasum)