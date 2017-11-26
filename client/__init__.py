"""
A simple wrapper around Deep Video Analytics REST API
"""
import os, json, requests, time, logging, base64

TYPE_QUERY_CONSTANT = 'Q'
TYPE_PROCESSING_CONSTANT = 'V'


class DVAContext(object):
    def __init__(self, server=None, token=None):
        if server:
            self.server = "{server}/api/".format(server=server)
        else:
            self.server = 'http://localhost:8000/api/'
        if token:
            self.token = token
        elif 'DVA_TOKEN' in os.environ:
            self.token = os.environ['DVA_TOKEN']
        elif os.path.isfile('creds.json'):
            self.token = json.loads(file('creds.json').read())['token']
        self.headers = {'Authorization': 'Token {}'.format(self.token)}

    def list_videos(self):
        r = requests.get("{server}/videos/".format(server=self.server),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            r.raise_for_status()

    def list_queries(self):
        r = requests.get("{server}/queries/".format(server=self.server),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            r.raise_for_status()

    def list_models(self):
        r = requests.get("{server}/models/".format(server=self.server),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            r.raise_for_status()

    def list_retrievers(self):
        r = requests.get("{server}/retrievers/".format(server=self.server),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            r.raise_for_status()

    def list_events(self):
        r = requests.get("{server}/events/".format(server=self.server),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            r.raise_for_status()

    def execute_query(self, query):
        r = requests.post("{server}/queries/".format(server=self.server), data={'script': json.dumps(query)},
                          headers=self.headers)
        if r.ok:
            return r.json()
        else:
            raise r.raise_for_status()

    def get_results(self, query_id):
        r = requests.get("{server}/queries/{query_id}/".format(server=self.server, query_id=query_id),
                         headers=self.headers)
        if r.ok:
            return r.json()
        else:
            raise r.raise_for_status()


class DVAQuery(object):
    def __init__(self):
        self.query_json = {}
        self.query_request = None
        self.context = None
        self.results = None
        self.query_id = None

    def execute(self, context):
        if self.query_request is None:
            self.query_request = context.execute_query(self.query_json)
            self.query_id = self.query_request['id']
            self.context = context
        else:
            raise ValueError("Query already requested")

    def wait(self, timeout=5, max_attempts=20):
        while not self.completed() and max_attempts > 0:
            logging.info("Query {qid} not completed sleeping for {timeout} and"
                         " waiting for at most {attempts} attempts, ".format(qid=self, timeout=timeout,
                                                                             attempts=max_attempts))
            max_attempts -= 1
            time.sleep(timeout)

    def completed(self):
        if (self.results is None) or (not self.results['completed']):
            self.results = self.context.get_results(self.query_id)
        else:
            return self.results['completed']

    def view_results(self):
        pass


class ProcessVideoURL(DVAQuery):
    def __init__(self, name, url):
        super(ProcessVideoURL, self).__init__()
        self.url = url
        self.name = name
        self.query_json = {
            "process_type": TYPE_PROCESSING_CONSTANT,
            "tasks": [
                {
                    "operation": "perform_import",
                    "arguments": {
                        "source": "URL",
                        "url": self.url,
                        "name": self.name,
                        "next_tasks": [
                            {
                                "operation": "perform_video_segmentation",
                                "arguments": {
                                    "next_tasks": [
                                        {
                                            "operation": "perform_video_decode",
                                            "arguments": {
                                                "segments_batch_size": 10,
                                                "rate": 30,
                                                "rescale": 0,
                                                "next_tasks": [
                                                    {
                                                        "operation": "perform_indexing",
                                                        "arguments": {
                                                            "index": "inception",
                                                            "target": "frames",
                                                            "filters": "__parent__"
                                                        }
                                                    },
                                                    {
                                                        "operation": "perform_detection",
                                                        "arguments": {
                                                            "filters": "__parent__",
                                                            "detector": "coco",
                                                            "next_tasks": [
                                                                {
                                                                    "operation": "perform_indexing",
                                                                    "arguments": {
                                                                        "index": "inception",
                                                                        "target": "regions",
                                                                        "filters": {
                                                                            "event_id": "__parent_event__",
                                                                            "w__gte": 50,
                                                                            "h__gte": 50
                                                                        }
                                                                    }
                                                                }
                                                            ]
                                                        }
                                                    },
                                                    {
                                                        "operation": "perform_detection",
                                                        "arguments": {
                                                            "filters": "__parent__",
                                                            "detector": "face",
                                                            "next_tasks": [
                                                                {
                                                                    "operation": "perform_indexing",
                                                                    "arguments": {
                                                                        "index": "facenet",
                                                                        "target": "regions",
                                                                        "filters": {
                                                                            "event_id": "__parent_event__"
                                                                        }
                                                                    }
                                                                }
                                                            ]
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                }
            ]
        }


class FindSimilarImages(DVAQuery):
    def __init__(self, query_image_path, retriever_pk, n=20):
        super(FindSimilarImages, self).__init__()
        self.query_image_path = query_image_path
        self.query_json = {
            'process_type': TYPE_QUERY_CONSTANT,
            'image_data_b64': base64.encodestring(file(self.query_image_path).read()),
            'tasks': [
                {
                    'operation': 'perform_indexing',
                    'arguments': {
                        'index': 'inception',
                        'target': 'query',
                        'next_tasks': [
                            {'operation': 'perform_retrieval',
                             'arguments': {'count': n, 'retriever_pk': retriever_pk}
                             }
                        ]
                    }

                }

            ]
        }