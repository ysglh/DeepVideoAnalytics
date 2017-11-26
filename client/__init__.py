"""
A simple wrapper around Deep Video Analytics REST API
"""
import os, json, requests


class DVAContext(object):

    def __init__(self,server=None,token=None):
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


class DVAQuery(object):

    def __init__(self):
        pass

    def to_json(self):
        pass

    def execute(self,context):
        pass

    def wait(self):
        pass