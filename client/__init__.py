"""
A simple wrapper around Deep Video Analytics REST API
"""


class DVAContext(object):

    def __init__(self,server=None,token=None):
        pass

    def list_videos(self):
        pass

    def list_queries(self):
        pass

    def list_models(self):
        pass

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