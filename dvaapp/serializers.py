from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, VLabel, Frame, Annotation, Detection, Query, QueryResults, TEvent, IndexEntries


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'password')
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

    def update(self, instance, validated_data):
        if 'password' in validated_data:
            password = validated_data.pop('password')
            instance.set_password(password)
        return super(UserSerializer, self).update(instance, validated_data)


class VideoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Video
        fields = '__all__'


class VLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VLabel
        fields = '__all__'


class FrameSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Frame
        fields = '__all__'


class DetectionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Detection
        fields = '__all__'


class AnnotationSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Annotation
        fields = '__all__'


class QuerySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Query
        fields = '__all__'


class QueryResultsSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = QueryResults
        fields = '__all__'


class TEventSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = TEvent
        fields = '__all__'


class IndexEntriesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = IndexEntries
        fields = '__all__'


class FrameExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Frame
        fields = '__all__'


class AnnotationExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Annotation
        fields = '__all__'


class DetectionExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Detection
        fields = '__all__'


class IndexEntryExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = IndexEntries
        fields = '__all__'


class VideoExportSerializer(serializers.ModelSerializer):
    frame_list = FrameExportSerializer(source='frame_set',read_only=True,many=True)
    annotation_list = AnnotationExportSerializer(source='annotation_set',read_only=True,many=True)
    detection_list = DetectionExportSerializer(source='detection_set',read_only=True,many=True)
    index_entries_list = IndexEntryExportSerializer(source='indexentries_set',read_only=True,many=True)

    class Meta:
        model = Video
        fields = ('name','length_in_seconds','height','width','metadata',
                  'frames','created','description','uploaded','dataset',
                  'uploader','detections','url','youtube_video','annotation_list',
                  'frame_list','detection_list','index_entries_list')
