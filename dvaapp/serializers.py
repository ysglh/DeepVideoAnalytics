from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, AppliedLabel, Frame, Region, DVAPQL, QueryResults, TEvent, IndexEntries, VDNDataset, VDNServer, Tube, Clusters, ClusterCodes, Segment
import os, json, logging, glob
from collections import defaultdict
from django.conf import settings
from StringIO import StringIO
from rest_framework.parsers import JSONParser

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


class TubeSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Tube
        fields = '__all__'


class ClustersSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Clusters
        fields = '__all__'


class ClusterCodesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ClusterCodes
        fields = '__all__'


class DVAPQLSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = DVAPQL
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
        fields = ('region_list','video','frame_index','keyframe','w','h','t','name','subdir','id','segment_index')


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


class TubeExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tube
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
    tube_list = TubeExportSerializer(source='tube_set', read_only=True, many=True)

    class Meta:
        model = Video
        fields = ('name','length_in_seconds','height','width','metadata','frames','created','description','uploaded','dataset',
                  'uploader','segments','url','youtube_video','frame_list','segment_list','event_list','label_list','tube_list','index_entries_list')


def create_region(a,video_obj,vdn_dataset,event_to_pk=None):
    da = Region()
    da.video = video_obj
    da.x = a['x']
    da.y = a['y']
    da.h = a['h']
    da.w = a['w']
    da.vdn_key = a['id']
    da.metadata_text = a['metadata_text']
    da.metadata_json = a['metadata_json']
    da.materialized = a.get('materialized',False)
    da.png = a.get('png',False)
    da.region_type = a['region_type']
    da.confidence = a['confidence']
    da.object_name = a['object_name']
    da.full_frame = a['full_frame']
    if a.get('event',None):
        da.event_id = event_to_pk[a['event']]
    da.parent_frame_index = a['parent_frame_index']
    da.parent_segment_index = a.get('parent_segment_index',-1)
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


def create_frame(f,video_obj):
    df = Frame()
    df.video = video_obj
    df.name = f['name']
    df.frame_index = f['frame_index']
    df.subdir = f['subdir']
    df.h = f.get('h',0)
    df.w = f.get('w',0)
    df.t = f.get('t',0)
    df.segment_index = f.get('segment_index',0)
    df.keyframe = f.get('keyframe',False)
    return df


def import_tubes(tubes,video_obj):
    """
    :param segments:
    :param video_obj:
    :return:
    """
    # TODO: Implement this
    raise NotImplementedError


def import_detector(dd):
    dd.phase_1_log = file("{}/detectors/{}/phase_1.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    dd.phase_2_log = file("{}/detectors/{}/phase_2.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    with open("{}/detectors/{}/input.json".format(settings.MEDIA_ROOT, dd.pk)) as fh:
        metadata = json.load(fh)
    if 'class_distribution' in metadata:
        dd.class_distribution = json.dumps(metadata['class_distribution'])
    else:
        dd.class_distribution = json.dumps(metadata['class_names'])
    dd.class_names = json.dumps(metadata['class_names'])


def create_event(e,v):
    de = TEvent()
    de.started = e.get('started',False)
    de.start_ts = e.get('start_ts',None)
    de.completed = e.get('completed',False)
    de.errored = e.get('errored',False)
    de.error_message = e.get('error_message',"")
    de.video_id = v.pk
    de.operation = e.get('operation',"")
    de.created = e['created']
    de.seconds = e.get('seconds',-1)
    de.arguments = e.get('arguments',{})
    de.task_id = e.get('task_id',"")
    return de


def create_segment(s,v):
    ds = Segment()
    ds.video_id = v.pk
    ds.segment_index = s.get('segment_index','-1')
    ds.start_time = s.get('start_time',0)
    ds.end_time = s.get('end_time',0)
    ds.metadata = s.get('metadata',"")
    ds.frame_count = s.get('frame_count',0)
    ds.start_index = s.get('start_index',0)
    return ds


class VideoImporter(object):
    
    def __init__(self,video,json,root_dir):
        self.video = video
        self.json = json
        self.root = root_dir
        self.region_to_pk = {}
        self.frame_to_pk = {}
        self.event_to_pk = {}
        self.segment_to_pk = {}

    def import_video(self):
        self.video.name = self.json['name']
        self.video.frames = self.json['frames']
        self.video.height = self.json['height']
        self.video.width = self.json['width']
        self.video.segments = self.json.get('segments',0)
        self.video.youtube_video = self.json['youtube_video']
        self.video.dataset = self.json['dataset']
        self.video.url = self.json['url']
        self.video.description = self.json['description']
        self.video.metadata = self.json['metadata']
        self.video.length_in_seconds = self.json['length_in_seconds']
        self.video.save()
        if not self.video.dataset:
            old_video_path = [fname for fname in glob.glob("{}/video/*.mp4".format(self.root))][0]
            new_video_path = "{}/video/{}.mp4".format(self.root, self.video.pk)
            os.rename(old_video_path, new_video_path)
        self.import_events()
        self.import_segments()
        self.bulk_import_frames()
        self.convert_regions_files()
        self.import_index_entries()

    def import_segments(self):
        old_ids = []
        segments = []
        for s in self.json.get('segment_list', []):
            old_ids.append(s['id'])
            segments.append(create_segment(s,self.video))
        segment_ids = Segment.objects.bulk_create(segments, 1000)
        for i, k in enumerate(segment_ids):
            self.segment_to_pk[old_ids[i]] = k.id

    def import_events(self):
        old_ids = []
        children_ids = defaultdict(list)
        events = []
        for e in self.json.get('event_list', []):
            old_ids.append(e['id'])
            if 'parent' in e:
                children_ids[e['parent']].append(e['id'])
            events.append(create_event(e,self.video))
        event_ids = TEvent.objects.bulk_create(events,1000)
        for i,k in enumerate(event_ids):
            self.event_to_pk[old_ids[i]] = k.id
        for old_id in old_ids:
            parent_id = self.event_to_pk[old_id]
            for child_old_id in children_ids[old_id]:
                ce = TEvent.objects.get(pk=self.event_to_pk[child_old_id])
                ce.parent_id = parent_id
                ce.save()
        if old_ids:
            # This is the export task that led to the video being exported and hence should be deleted
            last = TEvent.objects.get(pk=self.event_to_pk[max(old_ids)])
            last.seconds = 0
            last.completed = True
            last.save()

    def convert_regions_files(self):
        if os.path.isdir('{}/detections/'.format(self.root)):
            source_subdir = 'detections'  # temporary for previous version imports
            os.mkdir('{}/regions'.format(self.root))
        else:
            source_subdir = 'regions'
        convert_list = []
        for k, v in self.region_to_pk.iteritems():
            dd = Region.objects.get(pk=v)
            original = '{}/{}/{}.jpg'.format(self.root, source_subdir, k)
            temp_file = "{}/regions/d_{}.jpg".format(self.root, v)
            converted = "{}/regions/{}.jpg".format(self.root, v)
            if dd.materialized or os.path.isfile(original):
                try:
                    os.rename(original, temp_file)
                    convert_list.append((temp_file, converted))
                except:
                    raise ValueError, "could not copy {} to {}".format(original, temp_file)
        for temp_file, converted in convert_list:
            os.rename(temp_file, converted)

    def import_index_entries(self):
        previous_transformed = set()
        for i in self.json['index_entries_list']:
            di = IndexEntries()
            di.video = self.video
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
                entries = json.load(file('{}/indexes/{}'.format(self.root, di.entries_file_name)))
                transformed = []
                for entry in entries:
                    entry['video_primary_key'] = self.video.pk
                    if 'detection_primary_key' in entry:
                        entry['detection_primary_key'] = self.region_to_pk[entry['detection_primary_key']]
                    if 'frame_primary_key' in entry:
                        entry['frame_primary_key'] = self.frame_to_pk[entry['frame_primary_key']]
                    transformed.append(entry)
                with open('{}/indexes/{}'.format(self.root, di.entries_file_name), 'w') as output:
                    json.dump(transformed, output)
                di.save()

    def bulk_import_frames(self):
        vdn_dataset = self.video.vdn_dataset
        frame_regions = defaultdict(list)
        frames = []
        frame_index_to_fid = {}
        for i, f in enumerate(self.json['frame_list']):
            frames.append(create_frame(f, self.video))
            frame_index_to_fid[i] = f['id']
            if 'region_list' in f:
                for a in f['region_list']:
                    ra = create_region(a, self.video, vdn_dataset,self.event_to_pk)
                    if ra.region_type == Region.DETECTION:
                        frame_regions[i].append((ra, a['id']))
                    else:
                        frame_regions[i].append((ra, None))
            elif 'detection_list' in f or 'annotation_list' in f:
                raise NotImplementedError, "Older format no longer supported"
        bulk_frames = Frame.objects.bulk_create(frames)
        regions = []
        regions_index_to_rid = {}
        region_index = 0
        bulk_regions = []
        for i, k in enumerate(bulk_frames):
            self.frame_to_pk[frame_index_to_fid[i]] = k.id
            for r, rid in frame_regions[i]:
                r.frame_id = k.id
                regions.append(r)
                regions_index_to_rid[region_index] = rid
                region_index += 1
                if len(regions) == 1000:
                    bulk_regions.extend(Region.objects.bulk_create(regions))
                    regions = []
        bulk_regions.extend(Region.objects.bulk_create(regions))
        regions = []
        for i, k in enumerate(bulk_regions):
            if regions_index_to_rid[i]:
                self.region_to_pk[regions_index_to_rid[i]] = k.id