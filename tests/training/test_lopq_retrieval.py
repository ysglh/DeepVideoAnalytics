import os, glob, sys
import django
sys.path.append("../../server/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from dvaapp.operations import indexing, approximation, retrieval


if __name__ == '__main__':

    facenet, _ = indexing.Indexers.get_index_by_name('facenet')
    a, _ = approximation.Approximators.get_approximator_by_name('Facenet_LOPQ_on_LFW')
    entries = []
    for p in glob.glob(os.path.join(os.path.dirname('__file__'), "facenet_lopq/lopq_face_test/*.jpg")):
        entries.append(
            {
                'path': p,
                'codes': a.approximate(facenet.apply(p))
            }
        )
    r, _ = retrieval.Retrievers.get_retriever(3)
    r.load_index(entries=entries)
    query_vector = facenet.apply(os.path.join(os.path.dirname('__file__'), "facenet_lopq/query.jpg"))
    for k in r.nearest(query_vector)[:10]:
        print k