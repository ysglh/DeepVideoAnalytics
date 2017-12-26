#!/usr/bin/env python
import django, os, sys
sys.path.append('../server/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
import base64
from dvaapp.models import DVAPQL, Retriever, QueryResults
from dvaapp.processing import DVAPQLProcess


if __name__ == '__main__':
    query_dict = {
        'process_type': DVAPQL.QUERY,
        'image_data_b64': base64.encodestring(file('queries/query.png').read()),
        'tasks': [
            {
                'operation': 'perform_indexing',
                'arguments': {
                    'index': 'inception',
                    'target': 'query',
                    'map': [
                        {'operation': 'perform_retrieval',
                         'arguments': {'count': 15, 'retriever_pk': Retriever.objects.get(name='inception',
                                                                                          algorithm=Retriever.EXACT).pk}
                         }
                    ]
                }

            },
            {
                'operation': 'perform_detection',
                'arguments': {
                    'detector': 'coco',
                    'target': 'query',
                }

            }

        ]
    }
    qp = DVAPQLProcess()
    qp.create_from_json(query_dict)
    qp.launch()
    qp.wait(timeout=400)
    print QueryResults.objects.count()
