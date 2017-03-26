from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, FrameLabel, VLabel, Frame, Annotation, Detection, Query, QueryResults

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'is_staff')


class VideoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Video
        fields = ('name',
                  'height',
                  'width',
                  'length_in_seconds',
                  'created',
                  'description',
                  'uploaded',
                  'dataset',
                  'uploader',
                  'frames',
                  'detections',
                  'metadata',
                  'query',
                  'url',
                  'youtube_video',
                  'parent_query'
                  )


class FrameLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = FrameLabel
        fields = ('frame', 'video', 'label', 'source', 'label_parent', 'annotation')


class VLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VLabel
        fields = ('created','label_name')


class FrameSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Frame
        fields = ('video','name','frame_index','subdir')


class DetectionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Detection
        fields = ('video',
                  'object_name',
                  'frame',
                  'x',
                  'y',
                  'h',
                  'w',
                  'confidence',
                  'metadata')


class AnnotationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Annotation
        fields = ('video',
                  'name',
                  'frame',
                  'x',
                  'y',
                  'h',
                  'w',
                  'label_count',
                  'user',
                  'created',
                  'metadata_text')


class QuerySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Query
        fields = ('created',
                  'user',
                  'results',
                  'results_metadata',)


class QueryResultsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = QueryResults
        fields = ('query',
                  'video',
                  'frame',
                  'rank',
                  'algorithm',
                  'distance',
                  'detection',)