"""
A simple wrapper around Deep Video Analytics REST API
"""
import os, json, requests


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

    def list_events(self):
        pass

    def execute_query(self,query):
        r = requests.post(self.server, data={'script': json.dumps(query)}, headers=self.headers)
        if r.ok:
            return r.json()
        else:
            raise r.raise_for_status()


class DVAQuery(object):
    def __init__(self):
        self.query_json = {}
        self.query_request = None

    def execute(self, context):
        if self.query_request is None:
            self.query_request = context.execute_query(self.query_json)
        else:
            raise ValueError("Query already requested")

    def wait(self):
        pass

    def view_results(self):
        pass


class ProcessVideoURL(DVAQuery):
    def __init__(self, name, url):
        super(ProcessVideoURL, self).__init__()
        self.url = url
        self.name = name
        self.query_json = {
            "process_type": "V",
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
                                                                    "operation": "perform_transformation",
                                                                    "arguments": {
                                                                        "filters": {
                                                                            "event_id": "__parent_event__"
                                                                        },
                                                                        "next_tasks": [
                                                                            {
                                                                                "operation": "perform_indexing",
                                                                                "arguments": {
                                                                                    "index": "inception",
                                                                                    "target": "regions",
                                                                                    "filters": {
                                                                                        "event_id": "__grand_parent_event__",
                                                                                        "w__gte": 50,
                                                                                        "h__gte": 50
                                                                                    }
                                                                                }
                                                                            }
                                                                        ]
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
                                                                    "operation": "perform_transformation",
                                                                    "arguments": {
                                                                        "resize": [
                                                                            182,
                                                                            182
                                                                        ],
                                                                        "filters": {
                                                                            "event_id": "__parent_event__"
                                                                        },
                                                                        "next_tasks": [
                                                                            {
                                                                                "operation": "perform_indexing",
                                                                                "arguments": {
                                                                                    "index": "facenet",
                                                                                    "target": "regions",
                                                                                    "filters": {
                                                                                        "event_id": "__grand_parent_event__"
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
                }
            ]
        }
