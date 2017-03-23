from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'is_staff')


class VideoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Video
        fields = ('name', 'name', 'height', 'width', 'length_in_seconds')
