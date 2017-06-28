from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, AppliedLabel, Frame, Region, Query, QueryResults, TEvent, IndexEntries, VDNDataset, VDNServer, Scene, Clusters, ClusterCodes, Segment
import os, json, logging, glob
from collections import defaultdict
from django.conf import settings

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'password')
        extra_kwargs = {
            'password': {'write_only': True},
        }

    # def create(self, validated_data):
    #     user = User.objects.create_user(**validated_data)
    #     return user
    #
    # def update(self, instance, validated_data):
    #     if 'password' in validated_data:
    #         password = validated_data.pop('password')
    #         instance.set_password(password)
    #     return super(UserSerializer, self).update(instance, validated_data)


class VideoSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Video
        fields = '__all__'


class AppliedLabelSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = AppliedLabel
        fields = '__all__'


class FrameSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Frame
        fields = '__all__'


class SegmentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Segment
        fields = '__all__'


class RegionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Region
        fields = '__all__'


class SceneSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Scene
        fields = '__all__'


class ClustersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Clusters
        fields = '__all__'


class ClusterCodesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ClusterCodes
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


class RegionExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Region
        fields = '__all__'


class FrameExportSerializer(serializers.ModelSerializer):
    region_list = RegionExportSerializer(source='region_set',read_only=True,many=True)

    class Meta:
        model = Frame
        fields = ('region_list','video','frame_index','keyframe','t','name','subdir','id','segment_index')


class IndexEntryExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = IndexEntries
        fields = '__all__'


class TEventExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = TEvent
        fields = '__all__'


class AppliedLabelExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = AppliedLabel
        fields = '__all__'


class SceneExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scene
        fields = '__all__'


class SegmentExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Segment
        fields = '__all__'


class VideoExportSerializer(serializers.ModelSerializer):
    frame_list = FrameExportSerializer(source='frame_set',read_only=True,many=True)
    segment_list = SegmentExportSerializer(source='segment_set',read_only=True,many=True)
    index_entries_list = IndexEntryExportSerializer(source='indexentries_set',read_only=True,many=True)
    event_list = TEventExportSerializer(source='tevent_set',read_only=True,many=True)
    label_list = AppliedLabelExportSerializer(source='appliedlabel_set', read_only=True, many=True)
    scene_list = SceneExportSerializer(source='scene_set', read_only=True, many=True)

    class Meta:
        model = Video
        fields = ('name','length_in_seconds','height','width','metadata','frames','created','description','uploaded','dataset',
                  'uploader','segments','url','youtube_video','frame_list','segment_list','event_list','label_list','scene_list','index_entries_list')


def create_region(a,video_obj,vdn_dataset):
    da = Region()
    da.video = video_obj
    da.x = a['x']
    da.y = a['y']
    da.h = a['h']
    da.w = a['w']
    da.vdn_key = a['id']
    da.metadata_text = a['metadata_text']
    da.metadata_json = a['metadata_json']
    da.region_type = a['region_type']
    da.confidence = a['confidence']
    da.object_name = a['object_name']
    da.full_frame = a['full_frame']
    da.parent_frame_index = a['parent_frame_index']
    if 'parent_segment_index' in a:
        da.parent_frame_index = a['parent_segment_index']
    if vdn_dataset:
        da.vdn_dataset = vdn_dataset
    return da


def import_region(a,video_obj,frame,detection_to_pk,vdn_dataset=None):
    da = create_region(a,video_obj,vdn_dataset)
    da.frame = frame
    da.save()
    if da.region_type == Region.DETECTION:
        detection_to_pk[a['id']]=da.pk
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



def create_frame(f,video_obj):
    df = Frame()
    df.video = video_obj
    df.name = f['name']
    df.frame_index = f['frame_index']
    df.subdir = f['subdir']
    return df


def import_frame(f,video_obj,detection_to_pk,vdn_dataset=None):
    df = create_frame(f, video_obj)
    df.save()
    if 'region_list' in f:
        for a in f['region_list']:
            da = import_region(a,video_obj,df,detection_to_pk,vdn_dataset)
    elif 'detection_list' in f or 'annotation_list' in f:
            raise NotImplementedError, "Older format no longer supported"
    return df


def bulk_import_frames(flist, video_obj, frame_to_pk, detection_to_pk, vdn_dataset):
    frame_regions = defaultdict(list)
    frames = []
    frame_index_to_fid = {}
    for i,f in enumerate(flist):
        frames.append(create_frame(f, video_obj))
        frame_index_to_fid[i] = f['id']
        if 'region_list' in f:
            for a in f['region_list']:
                ra = create_region(a, video_obj, vdn_dataset)
                if ra.region_type == Region.DETECTION:
                    frame_regions[i].append((ra,a['id']))
                else:
                    frame_regions[i].append((ra, None))
        elif 'detection_list' in f or 'annotation_list' in f:
            raise NotImplementedError,"Older format no longer supported"
    bulk_frames = Frame.objects.bulk_create(frames)
    regions = []
    regions_index_to_rid = {}
    region_index = 0
    bulk_regions = []
    for i,k in enumerate(bulk_frames):
        frame_to_pk[frame_index_to_fid[i]] = k.id
        for r,rid in frame_regions[i]:
            r.frame_id = k.id
            regions.append(r)
            regions_index_to_rid[region_index] = rid
            region_index += 1
            if len(regions) == 1000:
                bulk_regions.extend(Region.objects.bulk_create(regions))
                regions = []
    bulk_regions.extend(Region.objects.bulk_create(regions))
    regions = []
    for i,k in enumerate(bulk_regions):
        if regions_index_to_rid[i]:
            detection_to_pk[regions_index_to_rid[i]] = k.id


def import_video_json(video_obj,video_json,video_root_dir):
    video_obj.name = video_json['name']
    video_obj.frames = video_json['frames']
    video_obj.height = video_json['height']
    video_obj.width = video_json['width']
    if 'segments' in video_json:
        video_obj.segments = video_json['segments']
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
    detection_to_pk, frame_to_pk = {}, {}
    bulk_import_frames(video_json['frame_list'], video_obj, frame_to_pk, detection_to_pk, vdn_dataset)
    for k,v in detection_to_pk.iteritems():
        original = '{}/detections/{}.jpg'.format(video_root_dir, k)
        temp_file = "{}/detections/d_{}.jpg".format(video_root_dir,v)
        os.rename(original, temp_file)
    for k, v in detection_to_pk.iteritems():
        temp_file = "{}/detections/d_{}.jpg".format(video_root_dir, v)
        converted = "{}/detections/{}.jpg".format(video_root_dir, v)
        os.rename(temp_file, converted)
    previous_transformed = set()
    for i in video_json['index_entries_list']:
        import_index_entries(i, video_obj, previous_transformed, detection_to_pk, frame_to_pk, video_root_dir)


def import_detector(dd):
    dd.phase_1_log = file("{}/models/{}/phase_1.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    dd.phase_2_log = file("{}/models/{}/phase_2.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    with open("{}/models/{}/input.json".format(settings.MEDIA_ROOT, dd.pk)) as fh:
        metadata = json.load(fh)
    if 'class_distribution' in metadata:
        dd.class_distribution = json.dumps(metadata['class_distribution'])
    else:
        dd.class_distribution = json.dumps(metadata['class_names'])
    dd.class_names = json.dumps(metadata['class_names'])