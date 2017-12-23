

class BaseRegionTransformer(object):

    def __init__(self,outputs_png):
        self.outputs_png = outputs_png

    def tranform_path(self,image_path):
        pass

    def tranform_image(self,im):
        pass


class BaseTubeTransformer(object):

    def __init__(self,outputs_video):
        self.outputs_video = outputs_video

    def transform_tube(self,tube):
        pass


class SemanticSegmentation(BaseRegionTransformer):

    def __init__(self, network_path, outputs_png):
        self.network_path = network_path
        super(SemanticSegmentation, self).__init__(outputs_png)