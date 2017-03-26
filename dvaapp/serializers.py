from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, FrameLabel, VLabel

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
                  'detections',
                  'metadata',
                  'query',
                  )


class FrameLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = FrameLabel
        fields = ('frame', 'video', 'label', 'source', 'label_parent', 'annotation')


class VLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VLabel
        fields = ('created','label_name')
