from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, VLabel, Frame, Annotation, Detection, Query, QueryResults, TEvent, IndexEntries, VDNDataset, VDNServer
import os, json, logging, glob


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


class VDNDatasetSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VDNDataset
        fields = '__all__'


class VDNServerSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = VDNServer
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

def import_frame(f,video_obj):
    df = Frame()
    df.video = video_obj
    df.name = f['name']
    df.frame_index = f['frame_index']
    df.subdir = f['subdir']
    df.save()
    return df


def import_detection(d,video_obj,frame_to_pk,vdn_dataset=None):
    dd = Detection()
    dd.video = video_obj
    dd.x = d['x']
    dd.y = d['y']
    dd.h = d['h']
    dd.w = d['w']
    dd.frame_id = frame_to_pk[d['frame']]
    dd.confidence = d['confidence']
    dd.object_name = d['object_name']
    dd.metadata = d['metadata']
    if vdn_dataset:
        dd.vdn_dataset = vdn_dataset
    dd.vdn_key = d['id']
    dd.save()
    return dd


def import_annotation(a,video_obj,frame_to_pk,detection_to_pk,vdn_dataset=None):
    da = Annotation()
    da.video = video_obj
    da.x = a['x']
    da.y = a['y']
    da.h = a['h']
    da.w = a['w']
    da.vdn_key = a['id']
    if vdn_dataset:
        da.vdn_dataset = vdn_dataset
    if a['label'].strip():
        da.label = a['label']
        if vdn_dataset:
            label_object, created = VLabel.objects.get_or_create(label_name=a['label'], source=VLabel.VDN, video=video_obj, vdn_dataset=vdn_dataset)
        else:
            label_object, created = VLabel.objects.get_or_create(label_name=a['label'], source=VLabel.UI, video=video_obj)
        da.label_parent = label_object
    da.frame_id = frame_to_pk[a['frame']]
    if a['detection']:
        da.detection_id = detection_to_pk[a['detection']]
    da.full_frame = a['full_frame']
    da.metadata_text = a['metadata_text']
    da.save()
    return da


def import_index_entries(i,video_obj,previous_transformed,detection_to_pk,frame_to_pk,video_root_dir):
    di = IndexEntries()
    di.video = video_obj
    di.algorithm = i['algorithm']
    di.count = i['count']
    di.contains_detections = i['contains_detections']
    di.contains_frames = i['contains_frames']
    di.approximate = i['approximate']
    di.created = i['created']
    di.features_file_name = i['features_file_name']
    di.entries_file_name = i['entries_file_name']
    di.detection_name = i['detection_name']
    signature = "{}".format(di.entries_file_name)
    if signature in previous_transformed:
        logging.warning("repeated index entries found, skipping {}".format(signature))
    else:
        previous_transformed.add(signature)
        transform_index_entries(di, detection_to_pk, frame_to_pk, video_obj.pk, video_root_dir)
        di.save()


def transform_index_entries(di,detection_to_pk,frame_to_pk,video_id,video_root_dir):
    entries = json.load(file('{}/indexes/{}'.format(video_root_dir, di.entries_file_name)))
    transformed = []
    for entry in entries:
        entry['video_primary_key'] = video_id
        if 'detection_primary_key' in entry:
            entry['detection_primary_key'] = detection_to_pk[entry['detection_primary_key']]
        if 'frame_primary_key' in entry:
            entry['frame_primary_key'] = frame_to_pk[entry['frame_primary_key']]
        transformed.append(entry)
    with open('{}/indexes/{}'.format(video_root_dir, di.entries_file_name),'w') as output:
        json.dump(transformed,output)


def import_video_json(video_obj,video_json,video_root_dir):
    video_obj.name = video_json['name']
    video_obj.frames = video_json['frames']
    video_obj.height = video_json['height']
    video_obj.width = video_json['width']
    video_obj.detections = video_json['detections']
    video_obj.youtube_video = video_json['youtube_video']
    video_obj.dataset = video_json['dataset']
    video_obj.url = video_json['url']
    video_obj.description = video_json['description']
    video_obj.metadata = video_json['metadata']
    video_obj.length_in_seconds = video_json['length_in_seconds']
    video_obj.save()
    vdn_dataset = video_obj.vdn_dataset
    if not video_obj.dataset:
        old_video_path = [fname for fname in glob.glob("{}/video/*.mp4".format(video_root_dir))][0]
        new_video_path = "{}/video/{}.mp4".format(video_root_dir,video_obj.pk)
        os.rename(old_video_path,new_video_path)
    frame_to_pk = {}
    detection_to_pk = {}
    for f in video_json['frame_list']:
        df = import_frame(f,video_obj)
        frame_to_pk[f['id']] = df.pk
    for d in video_json['detection_list']:
        dd = import_detection(d,video_obj,frame_to_pk,vdn_dataset)
        detection_to_pk[d['id']]=dd.pk
    for k,v in detection_to_pk.iteritems():
        original = '{}/detections/{}.jpg'.format(video_root_dir, k)
        temp_file = "{}/detections/d_{}.jpg".format(video_root_dir,v)
        os.rename(original, temp_file)
    for k, v in detection_to_pk.iteritems():
        temp_file = "{}/detections/d_{}.jpg".format(video_root_dir, v)
        converted = "{}/detections/{}.jpg".format(video_root_dir, v)
        os.rename(temp_file, converted)
    for a in video_json['annotation_list']:
        da = import_annotation(a,video_obj,frame_to_pk,detection_to_pk,vdn_dataset)
    previous_transformed = set()
    for i in video_json['index_entries_list']:
        import_index_entries(i, video_obj, previous_transformed, detection_to_pk, frame_to_pk, video_root_dir)
    os.remove("{}/{}.zip".format(video_root_dir, video_obj.pk))
