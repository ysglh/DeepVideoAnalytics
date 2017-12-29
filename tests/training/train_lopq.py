import os, sys, shutil
import django
sys.path.append("../../server/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
from dvaapp.models import TrainedModel
from dvaapp.operations import indexing
from dvalib.trainers import lopq_trainer
from dvalib.retriever import LOPQRetriever
from dvalib.indexer import LOPQIndexer
import numpy as np


if __name__ == '__main__':
    l = lopq_trainer.LOPQTrainer(name="Facenet LOPQ trained on LFW",
                                 dirname=os.path.join(os.path.dirname('__file__'),"facenet_lopq/"),
                                 components=32,m=16,v=16,sub=128,
                                 source_indexer_shashum="9f99caccbc75dcee8cb0a55a0551d7c5cb8a6836")
    data = np.load('facenet.npy')
    print data.shape
    l.train(data)
    j = l.save()
    m = TrainedModel(**j)
    m.save()
    m.create_directory()
    for f in m.files:
        shutil.copy(f['url'],'{}/models/{}/{}'.format(settings.MEDIA_ROOT,m.pk,f['filename']))
    facenet, _ = indexing.Indexers.get_index_by_name('facenet')
    i = LOPQIndexer(name=m.name,dirname="{}/models/{}/".format(settings.MEDIA_ROOT,m.pk),source_model=facenet)
    print i.approximate(data[0])
    print i.apply('../../docs/figures/leo.jpg')