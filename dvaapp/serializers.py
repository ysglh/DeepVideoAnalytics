from rest_framework import serializers, viewsets
from django.contrib.auth.models import User
from models import Video, Frame, Region, DVAPQL, QueryResults, TEvent, IndexEntries, \
    Tube, LOPQCodes, Segment, Label, VideoLabel, FrameLabel, RegionLabel, \
    SegmentLabel, TubeLabel, DeepModel, Retriever, SystemState, QueryRegion,\
    QueryRegionResults, Worker
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
    id = serializers.ReadOnlyField()

    class Meta:
        model = Video
        fields = '__all__'


class RetrieverSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = Retriever
        fields = '__all__'


class DeepModelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = DeepModel
        fields = '__all__'


class LabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = Label
        fields = '__all__'


class FrameLabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = FrameLabel
        fields = '__all__'


class RegionLabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = RegionLabel
        fields = '__all__'


class SegmentLabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = SegmentLabel
        fields = '__all__'


class VideoLabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = VideoLabel
        fields = '__all__'


class TubeLabelSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = TubeLabel
        fields = '__all__'


class FrameLabelExportSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = FrameLabel
        fields = '__all__'


class RegionLabelExportSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = RegionLabel
        fields = '__all__'


class SegmentLabelExportSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = SegmentLabel
        fields = '__all__'


class VideoLabelExportSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = VideoLabel
        fields = '__all__'


class WorkerSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Worker
        fields = ('queue_name', 'id')


class TubeLabelExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = TubeLabel
        fields = '__all__'


class FrameSerializer(serializers.HyperlinkedModelSerializer):
    media_url = serializers.SerializerMethodField()

    def get_media_url(self,obj):
        return "{}{}/frames/{}.jpg".format(settings.MEDIA_URL,obj.video_id,obj.frame_index)

    class Meta:
        model = Frame
        fields = ('url','media_url', 'video', 'frame_index', 'keyframe', 'w', 'h', 't',
                  'name', 'subdir', 'id', 'segment_index')


class SegmentSerializer(serializers.HyperlinkedModelSerializer):
    media_url = serializers.SerializerMethodField()

    def get_media_url(self,obj):
        return "{}{}/segments/{}.mp4".format(settings.MEDIA_URL,obj.video_id,obj.segment_index)

    class Meta:
        model = Segment
        fields = ('video','segment_index','start_time','end_time','metadata',
                  'frame_count','start_index','start_frame','end_frame','url','media_url', 'id')


class RegionSerializer(serializers.HyperlinkedModelSerializer):
    media_url = serializers.SerializerMethodField()

    def get_media_url(self,obj):
        if obj.materialized:
            return "{}{}/regions/{}.jpg".format(settings.MEDIA_URL,obj.video_id,obj.pk)
        else:
            return None


    class Meta:
        model = Region
        fields = ('url','media_url','region_type','video','user','frame','event','frame_index',
                  'segment_index','text','metadata','full_frame','x','y','h','w',
                  'polygon_points','created','object_name','confidence','materialized','png', 'id')


class TubeSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = Tube
        fields = '__all__'


class QueryRegionSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = QueryRegion
        fields = '__all__'


class LOPQCodesSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = LOPQCodes
        fields = '__all__'


class SystemStateSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SystemState
        fields = '__all__'


class QueryResultsSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = QueryResults
        fields = '__all__'


class QueryRegionResultsSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = QueryRegionResults
        fields = '__all__'


class QueryResultsExportSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = QueryResults
        fields = '__all__'


class QueryRegionResultsExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryRegionResults
        fields = '__all__'


class QueryRegionExportSerializer(serializers.ModelSerializer):
    query_region_results = QueryRegionResultsExportSerializer(source='queryregionresults_set', read_only=True, many=True)

    class Meta:
        model = QueryRegion
        fields = ('id','region_type','query','event','text','metadata','full_frame','x','y','h','w','polygon_points',
                  'created','object_name','confidence','png','query_region_results')


class TaskExportSerializer(serializers.ModelSerializer):
    query_results = QueryResultsExportSerializer(source='queryresults_set', read_only=True, many=True)
    query_regions = QueryRegionExportSerializer(source='queryregion_set', read_only=True, many=True)

    class Meta:
        model = TEvent
        fields = ('started','completed','errored','worker','error_message','video','operation','queue',
                  'created','start_ts','duration','arguments','task_id','parent','parent_process',
                  'imported','query_results', 'query_regions', 'id')


class TEventSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = TEvent
        fields = '__all__'


class IndexEntriesSerializer(serializers.HyperlinkedModelSerializer):
    id = serializers.ReadOnlyField()

    class Meta:
        model = IndexEntries
        fields = '__all__'


class RegionExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Region
        fields = '__all__'


class FrameExportSerializer(serializers.ModelSerializer):
    region_list = RegionExportSerializer(source='region_set', read_only=True, many=True)

    class Meta:
        model = Frame
        fields = ('region_list', 'video', 'frame_index', 'keyframe', 'w', 'h', 't',
                  'name', 'subdir', 'id', 'segment_index')


class IndexEntryExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = IndexEntries
        fields = '__all__'


class TEventExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = TEvent
        fields = '__all__'


class TubeExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tube
        fields = '__all__'


class SegmentExportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Segment
        fields = '__all__'


class DVAPQLSerializer(serializers.HyperlinkedModelSerializer):
    tasks = TaskExportSerializer(source='tevent_set', read_only=True, many=True)

    class Meta:
        model = DVAPQL
        fields =('process_type', 'created', 'user', 'image_data', 'script', 'tasks',
                 'results_metadata', 'results_available', 'completed','id')


class VideoExportSerializer(serializers.ModelSerializer):
    frame_list = FrameExportSerializer(source='frame_set', read_only=True, many=True)
    segment_list = SegmentExportSerializer(source='segment_set', read_only=True, many=True)
    index_entries_list = IndexEntryExportSerializer(source='indexentries_set', read_only=True, many=True)
    event_list = TEventExportSerializer(source='tevent_set', read_only=True, many=True)
    tube_list = TubeExportSerializer(source='tube_set', read_only=True, many=True)
    frame_label_list = FrameLabelExportSerializer(source='framelabel_set', read_only=True, many=True)
    region_label_list = RegionLabelExportSerializer(source='regionlabel_set', read_only=True, many=True)
    tube_label_list = TubeLabelExportSerializer(source='tubelabel_set', read_only=True, many=True)
    segment_label_list = SegmentLabelExportSerializer(source='segmentlabel_set', read_only=True, many=True)
    video_label_list = VideoLabelExportSerializer(source='videolabel_set', read_only=True, many=True)

    class Meta:
        model = Video
        fields = ('name', 'length_in_seconds', 'height', 'width', 'metadata', 'frames', 'created', 'description',
                  'uploaded', 'dataset', 'uploader', 'segments', 'url', 'youtube_video', 'frame_list', 'segment_list',
                  'event_list', 'tube_list', 'index_entries_list', 'frame_label_list', 'region_label_list',
                  'tube_label_list', 'segment_label_list', 'video_label_list')


def serialize_video_labels(v):
    serialized_labels = {}
    sources = [FrameLabel.objects.filter(video_id=v.pk), VideoLabel.objects.filter(video_id=v.pk),
               SegmentLabel.objects.filter(video_id=v.pk), RegionLabel.objects.filter(video_id=v.pk),
               TubeLabel.objects.filter(video_id=v.pk)]
    for source in sources:
        for k in source:
            if k.label_id not in serialized_labels:
                serialized_labels[k.label_id] = {'id':k.label.id,'name':k.label.name,'set':k.label.set}
    return serialized_labels.values()


def import_detector(dd):
    dd.phase_1_log = file("{}/detectors/{}/phase_1.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    dd.phase_2_log = file("{}/detectors/{}/phase_2.log".format(settings.MEDIA_ROOT, dd.pk)).read()
    with open("{}/models/{}/input.json".format(settings.MEDIA_ROOT, dd.pk)) as fh:
        metadata = json.load(fh)
    if 'class_distribution' in metadata:
        dd.class_distribution = json.dumps(metadata['class_distribution'])
    else:
        dd.class_distribution = json.dumps(metadata['class_names'])
    dd.class_names = json.dumps(metadata['class_names'])
    dd.save()


def import_frame_json(f,frame_index,event_id,video_id):
    regions = []
    df = Frame()
    df.video_id = video_id
    df.event_id = event_id
    df.frame_index = frame_index
    df.name = f['path']
    for r in f.get('regions',[]):
        regions.append(import_region_json(r,frame_index,video_id,event_id))
    return df,regions


def import_region_json(r,frame_index,video_id,event_id,segment_index=None,frame_id=None):
    dr = Region()
    dr.frame_index = frame_index
    dr.video_id = video_id
    dr.event_id = event_id
    dr.object_name = r['object_name']
    dr.region_type = r.get('region_type', Region.ANNOTATION)
    dr.full_frame = r.get('full_frame', False)
    if segment_index:
        dr.segment_index = segment_index
    if frame_id:
        dr.frame_id = frame_id
    dr.x = r.get('x', 0)
    dr.y = r.get('y', 0)
    dr.w = r.get('w', 0)
    dr.h = r.get('h', 0)
    dr.confidence = r.get('confidence', 0.0)
    if r.get('text', None):
        dr.text = r['text']
    else:
        dr.text = ""
    dr.metadata = r.get('metadata', None)
    return dr


def create_event(e, v):
    de = TEvent()
    de.imported = True
    de.started = e.get('started', False)
    de.start_ts = e.get('start_ts', None)
    de.completed = e.get('completed', False)
    de.errored = e.get('errored', False)
    de.error_message = e.get('error_message', "")
    de.video_id = v.pk
    de.operation = e.get('operation', "")
    de.created = e['created']
    if 'seconds' in e:
        de.duration = e.get('seconds', -1)
    else:
        de.duration = e.get('duration', -1)
    de.arguments = e.get('arguments', {})
    de.task_id = e.get('task_id', "")
    return de


class VideoImporter(object):
    def __init__(self, video, json, root_dir):
        self.video = video
        self.json = json
        self.root = root_dir
        self.region_to_pk = {}
        self.frame_to_pk = {}
        self.event_to_pk = {}
        self.segment_to_pk = {}
        self.label_to_pk = {}
        self.tube_to_pk = {}
        self.name_to_shasum = {'inception':'48b026cf77dfbd5d9841cca3ee550ef0ee5a0751',
                               'facenet':'9f99caccbc75dcee8cb0a55a0551d7c5cb8a6836',
                               'vgg':'52723231e796dd06fafd190957c8a3b5a69e009c'}

    def import_video(self):
        self.video.name = self.json['name']
        self.video.frames = self.json['frames']
        self.video.height = self.json['height']
        self.video.width = self.json['width']
        self.video.segments = self.json.get('segments', 0)
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
        self.import_labels()
        self.import_region_labels()
        self.import_frame_labels()
        self.import_segment_labels()
        self.import_tube_labels()
        self.import_video_labels()

    def import_labels(self):
        for l in self.json.get('labels', []):
            dl, _ = Label.objects.get_or_create(name=l['name'],set=l.get('set',''))
            self.label_to_pk[l['id']] = dl.pk

    def import_region_labels(self):
        region_labels = []
        for rl in self.json.get('region_label_list', []):
            drl = RegionLabel()
            drl.frame_id = self.frame_to_pk[rl['frame']]
            drl.region_id = self.region_to_pk[rl['region']]
            drl.video_id = self.video.pk
            if 'event' in rl:
                drl.event_id = self.event_to_pk[rl['event']]
            drl.frame_index = rl['frame_index']
            drl.segment_index = rl['segment_index']
            drl.label_id = self.label_to_pk[rl['label']]
            region_labels.append(drl)
        RegionLabel.objects.bulk_create(region_labels,1000)

    def import_frame_labels(self):
        frame_labels = []
        for fl in self.json.get('frame_label_list', []):
            dfl = FrameLabel()
            dfl.frame_id = self.frame_to_pk[fl['frame']]
            dfl.video_id = self.video.pk
            if 'event' in fl:
                dfl.event_id = self.event_to_pk[fl['event']]
            dfl.frame_index = fl['frame_index']
            dfl.segment_index = fl['segment_index']
            dfl.label_id = self.label_to_pk[fl['label']]
            frame_labels.append(dfl)
        FrameLabel.objects.bulk_create(frame_labels,1000)

    def import_segment_labels(self):
        segment_labels = []
        for sl in self.json.get('segment_label_list', []):
            dsl = SegmentLabel()
            dsl.video_id = self.video.pk
            if 'event' in sl:
                dsl.event_id = self.event_to_pk[sl['event']]
            dsl.segment_id = self.segment_to_pk[sl['segment']]
            dsl.segment_index = sl['segment_index']
            dsl.label_id = self.label_to_pk[sl['label']]
            segment_labels.append(dsl)
        SegmentLabel.objects.bulk_create(segment_labels,1000)

    def import_video_labels(self):
        video_labels = []
        for vl in self.json.get('video_label_list', []):
            dvl = VideoLabel()
            dvl.video_id = self.video.pk
            if 'event' in vl:
                dvl.event_id = self.event_to_pk[vl['event']]
            dvl.label_id = self.label_to_pk[vl['label']]
            video_labels.append(dvl)
        VideoLabel.objects.bulk_create(video_labels,1000)

    def import_tube_labels(self):
        tube_labels = []
        for tl in self.json.get('tube_label_list', []):
            dtl = TubeLabel()
            dtl.video_id = self.video.pk
            if 'event' in tl:
                dtl.event_id = self.event_to_pk[tl['event']]
            dtl.label_id = self.label_to_pk[tl['label']]
            dtl.tube_id = self.tube_to_pk[tl['tube']]
            tube_labels.append(dtl)
        TubeLabel.objects.bulk_create(tube_labels,1000)


    def import_segments(self):
        old_ids = []
        segments = []
        for s in self.json.get('segment_list', []):
            old_ids.append(s['id'])
            segments.append(self.create_segment(s))
        segment_ids = Segment.objects.bulk_create(segments, 1000)
        for i, k in enumerate(segment_ids):
            self.segment_to_pk[old_ids[i]] = k.id

    def create_segment(self,s):
        ds = Segment()
        ds.video_id = self.video.pk
        ds.segment_index = s.get('segment_index', '-1')
        ds.start_time = s.get('start_time', 0)
        ds.end_time = s.get('end_time', 0)
        ds.metadata = s.get('metadata', "")
        if s.get('event', None):
            ds.event_id = self.event_to_pk[s['event']]
        ds.frame_count = s.get('frame_count', 0)
        ds.start_index = s.get('start_index', 0)
        return ds

    def import_events(self):
        old_ids = []
        children_ids = defaultdict(list)
        events = []
        for e in self.json.get('event_list', []):
            old_ids.append(e['id'])
            if 'parent' in e:
                children_ids[e['parent']].append(e['id'])
            events.append(create_event(e, self.video))
        event_ids = TEvent.objects.bulk_create(events, 1000)
        for i, k in enumerate(event_ids):
            self.event_to_pk[old_ids[i]] = k.id
        for old_id in old_ids:
            parent_id = self.event_to_pk[old_id]
            for child_old_id in children_ids[old_id]:
                ce = TEvent.objects.get(pk=self.event_to_pk[child_old_id])
                ce.parent_id = parent_id
                ce.save()
        if 'export_event_pk' in self.json:
            last = TEvent.objects.get(pk=self.event_to_pk[self.json['export_event_pk']])
            last.duration = 0
            last.completed = True
            last.save()
        else:
            # Unless specified this is the export task that led to the video being exported
            #  and hence should be marked as completed
            last = TEvent.objects.get(pk=self.event_to_pk[max(old_ids)])
            last.duration = 0
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
            # defaults only for backward compatibility
            di.indexer_shasum =i.get('indexer_shasum',self.name_to_shasum[i['algorithm']])
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
        frame_regions = defaultdict(list)
        frames = []
        frame_index_to_fid = {}
        for i, f in enumerate(self.json['frame_list']):
            frames.append(self.create_frame(f))
            frame_index_to_fid[i] = f['id']
            if 'region_list' in f:
                for a in f['region_list']:
                    ra = self.create_region(a)
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

    def create_region(self, a):
        da = Region()
        da.video_id = self.video.pk
        da.x = a['x']
        da.y = a['y']
        da.h = a['h']
        da.w = a['w']
        da.vdn_key = a['id']
        if 'text' in a:
            da.text = a['text']
        elif 'metadata_text' in a:
            da.text = a['metadata_text']
        if 'metadata' in a:
            da.metadata = a['metadata']
        elif 'metadata_json' in a:
            da.metadata = a['metadata_json']
        da.materialized = a.get('materialized', False)
        da.png = a.get('png', False)
        da.region_type = a['region_type']
        da.confidence = a['confidence']
        da.object_name = a['object_name']
        da.full_frame = a['full_frame']
        if a.get('event', None):
            da.event_id = self.event_to_pk[a['event']]
        if 'parent_frame_index' in a:
            da.frame_index = a['parent_frame_index']
        else:
            da.frame_index = a['frame_index']
        if 'parent_segment_index' in a:
            da.segment_index = a.get('parent_segment_index', -1)
        else:
            da.segment_index = a.get('segment_index', -1)
        return da

    def create_frame(self, f):
        df = Frame()
        df.video_id = self.video.pk
        df.name = f['name']
        df.frame_index = f['frame_index']
        df.subdir = f['subdir']
        df.h = f.get('h', 0)
        df.w = f.get('w', 0)
        df.t = f.get('t', 0)
        if f.get('event', None):
            df.event_id = self.event_to_pk[f['event']]
        df.segment_index = f.get('segment_index', 0)
        df.keyframe = f.get('keyframe', False)
        return df

    def import_tubes(tubes, video_obj):
        """
        :param segments:
        :param video_obj:
        :return:
        """
        # TODO: Implement this
        raise NotImplementedError
