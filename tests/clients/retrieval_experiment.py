import requests
import os
import json
import sys
import base64

if __name__ == '__main__':
    if len(sys.argv) > 1:
        url,path = sys.argv[1],sys.argv[2]
    else:
        url = 'http://localhost:8199'
        path = '../queries/query.png'
    query_dict = {
        'process_type': 'Q',
        'image_data_b64': base64.encodestring(file(path).read()),
        'tasks': [
            {
                'operation': 'perform_indexing',
                'arguments': {
                    'index': 'inception',
                    'target': 'query',
                    'next_tasks': [
                        {'operation': 'perform_retrieval',
                         'arguments': {'count': 15, }
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
    if not os.path.isfile('creds.json'):
        raise ValueError,"Creds.json not found"
    token = json.loads(file('creds.json').read())['token']
    headers = {'Authorization': 'Token {}'.format(token)}
    r = requests.post("{}/api/queries/".format(url),
                      data={'script': json.dumps(query_dict)},
                      headers=headers)
    print r.json()
    print r.status_code
