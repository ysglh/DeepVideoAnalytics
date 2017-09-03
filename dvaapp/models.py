from __future__ import unicode_literals
import os, json
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField
from django.conf import settings


class VDNServer(models.Model):
    """
    A VDN server
    """
    url = models.URLField()
    name = models.CharField(max_length=200)
    last_response_datasets = JSONField(blank=True,null=True)
    last_response_detectors = JSONField(blank=True,null=True)
    last_token = models.CharField(max_length=300, default="")


class DVAPQL(models.Model):
    """
    A query object with image_data, can have multiple children subspecies
    """
    SCHEDULE = 'S'
    PROCESS = 'V'
    QUERY = 'Q'
    TYPE_CHOICES = ((SCHEDULE, 'Schedule'), (PROCESS, 'Process'), (QUERY, 'Query'))
    process_type = models.CharField(max_length=1, choices=TYPE_CHOICES, default=QUERY, )
    created = models.DateTimeField('date created', auto_now_add=True)
    user = models.ForeignKey(User, null=True, related_name="submitter")
    image_data = models.BinaryField(null=True)
    script = JSONField(blank=True, null=True)
    results_metadata = models.TextField(default="")
    results_available = models.BooleanField(default=False)



class Video(models.Model):
    name = models.CharField(max_length=500,default="")
    length_in_seconds = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    width = models.IntegerField(default=0)
    metadata = models.TextField(default="")
    frames = models.IntegerField(default=0)
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    uploaded = models.BooleanField(default=False)
    dataset = models.BooleanField(default=False)
    uploader = models.ForeignKey(User,null=True)
    segments = models.IntegerField(default=0)
    url = models.TextField(default="")
    youtube_video = models.BooleanField(default=False)
    parent_process = models.ForeignKey(DVAPQL,null=True)

    def __unicode__(self):
        return u'{}'.format(self.name)

    def path(self,media_root=None):
        if media_root:
            return "{}/{}/video/{}.mp4".format(media_root, self.pk, self.pk)
        else:
            return "{}/{}/video/{}.mp4".format(settings.MEDIA_ROOT,self.pk,self.pk)

    def create_directory(self, create_subdirs=True):
        os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, self.pk))
        if create_subdirs:
            os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/segments/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/regions/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/transforms/'.format(settings.MEDIA_ROOT, self.pk))
            os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, self.pk))


class IngestEntry(models.Model):
    video = models.ForeignKey(Video)
    ingest_index = models.IntegerField()
    ingest_filename = models.CharField(max_length=500)
    start_segment_index = models.IntegerField(null=True)
    start_frame_index = models.IntegerField(null=True)
    segments = models.IntegerField(null=True)
    frames = models.IntegerField(null=True)
    created = models.DateTimeField('date created', auto_now_add=True)

    class Meta:
        unique_together = (("video", "ingest_filename","ingest_index"),)


class TEvent(models.Model):
    started = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)
    errored = models.BooleanField(default=False)
    error_message = models.TextField(default="")
    video = models.ForeignKey(Video, null=True)
    operation = models.CharField(max_length=100, default="")
    queue = models.CharField(max_length=100, default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    start_ts = models.DateTimeField('date started', null=True)
    duration = models.FloatField(default=-1)
    arguments = JSONField(blank=True,null=True)
    task_id = models.TextField(null=True)
    parent = models.ForeignKey('self',null=True)
    parent_process = models.ForeignKey(DVAPQL,null=True)
    imported = models.BooleanField(default=False)


class Indexer(models.Model):
    """
    An indexer that can be used with any TF (eventually pytorch) network
    """
    TENSORFLOW = 'T'
    CAFFE = 'C'
    PYTORCH = 'P'
    OPENCV = 'O'
    MODES = (
        (TENSORFLOW, 'Tensorflow'),
        (CAFFE, 'Caffe'),
        (PYTORCH, 'Pytorch'),
        (OPENCV, 'OpenCV'),
    )
    mode = models.CharField(max_length=1,choices=MODES,db_index=True,default=TENSORFLOW)
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    shasum = models.CharField(max_length=40,null=True,unique=True)
    model_filename = models.CharField(max_length=200,default="")
    input_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_size = models.CharField(max_length=300,default="")
    created = models.DateTimeField('date created', auto_now_add=True)


class Retriever(models.Model):
    """
    # train_fraction = models.FloatField(default=0.8) # by default use 80% of data for training
    # cluster_count = models.IntegerField(default=0)
    # pca_file_name = models.CharField(max_length=200,default="")
    # model_file_name = models.CharField(max_length=200, default="")
    # components = models.IntegerField(default=64) # computer 64 principal components
    # started = models.DateTimeField('date created', auto_now_add=True)
    # completed = models.BooleanField(default=False)
    # m = models.IntegerField(default=16)
    # v = models.IntegerField(default=16)
    # sub = models.IntegerField(default=256)
    """
    EXACT = 'E'
    LOPQ = 'L'
    MODES = (
        (LOPQ, 'LOPQ'),
        (EXACT, 'Exact'),
    )
    algorithm = models.CharField(max_length=1,choices=MODES,db_index=True,default=EXACT)
    name = models.CharField(max_length=200,default="")
    arguments = JSONField(blank=True,null=True)
    source_filters = JSONField()
    created = models.DateTimeField('date created', auto_now_add=True)
    last_built = models.DateTimeField(null=True)

    def create_directory(self):
        os.mkdir('{}/retrievers/{}'.format(settings.MEDIA_ROOT, self.pk))

    def path(self):
        return '{}/retrievers/{}/'.format(settings.MEDIA_ROOT, self.pk)

    def proto_filename(self):
        return "{}/{}.proto".format(self.path(), self.pk)


class Analyzer(models.Model):
    """
    """
    TENSORFLOW = 'T'
    CAFFE = 'C'
    PYTORCH = 'P'
    OPENCV = 'O'
    MODES = (
        (TENSORFLOW, 'Tensorflow'),
        (CAFFE, 'Caffe'),
        (PYTORCH, 'Pytorch'),
        (OPENCV, 'OpenCV'),
    )
    mode = models.CharField(max_length=1,choices=MODES,db_index=True,default=TENSORFLOW)
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    model_filename = models.CharField(max_length=200,default="")
    produces_labels = models.BooleanField(default=False)
    produces_json = models.BooleanField(default=False)
    produces_text = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)


class Detector(models.Model):
    TENSORFLOW = 'T'
    CAFFE = 'C'
    PYTORCH = 'P'
    OPENCV = 'O'
    YOLO = 'Y'
    TFD = 'T'
    MODES = (
        (TENSORFLOW, 'Tensorflow'),
        (CAFFE, 'Caffe'),
        (PYTORCH, 'Pytorch'),
        (OPENCV, 'OpenCV'),
    )
    DETECTOR_TYPES = (
        (TFD, 'Tensorflow'),
        (YOLO, 'YOLO V2'),
    )
    mode = models.CharField(max_length=1,choices=MODES,db_index=True,default=TENSORFLOW)
    detector_type = models.CharField(max_length=1,choices=DETECTOR_TYPES,db_index=True,null=True)
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    model_filename = models.CharField(max_length=200,null=True)
    arguments = models.TextField(default="")
    phase_1_log = models.TextField(default="")
    phase_2_log = models.TextField(default="")
    class_distribution = models.TextField(default="")
    class_names = models.TextField(default="")
    frames_count = models.IntegerField(default=0)
    boxes_count = models.IntegerField(default=0)
    source = models.ForeignKey(TEvent, null=True)
    trained = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)
    class_index_to_string = JSONField(null=True,blank=True)

    def create_directory(self,create_subdirs=True):
        try:
            os.mkdir('{}/detectors/{}'.format(settings.MEDIA_ROOT, self.pk))
        except:
            pass

    def get_model_path(self,root_dir=None):
        if root_dir is None:
            root_dir = settings.MEDIA_ROOT
        return "{}/detectors/{}/{}".format(root_dir,self.pk,self.model_filename)

    def get_yolo_args(self):
        model_dir = "{}/detectors/{}/".format(settings.MEDIA_ROOT, self.pk)
        class_names = {k: v for k, v in json.loads(self.class_names)}
        args = {'root_dir': model_dir,
                'detector_pk': self.pk,
                'class_names':{i: k for k, i in class_names.items()}
                }
        return args

    def get_class_dist(self):
        return json.loads(self.class_distribution) if self.class_distribution.strip() else {}


class QueryIndexVector(models.Model):
    event = models.OneToOneField(TEvent)
    vector = models.BinaryField()
    created = models.DateTimeField('date created', auto_now_add=True)


class Frame(models.Model):
    video = models.ForeignKey(Video)
    frame_index = models.IntegerField()
    name = models.CharField(max_length=200,null=True)
    subdir = models.TextField(default="") # Retains information if the source is a dataset for labeling
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    t = models.FloatField(null=True) # time in seconds for keyframes
    keyframe = models.BooleanField(default=False) # is this a key frame for a video?
    segment_index = models.IntegerField(null=True)

    class Meta:
        unique_together = (("video", "frame_index"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.frame_index)

    def path(self,media_root=None):
        if media_root:
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,self.video_id,self.frame_index)


class Segment(models.Model):
    """
    A video segment useful for parallel dense decoding+processing as well as streaming
    """
    video = models.ForeignKey(Video)
    segment_index = models.IntegerField()
    start_time = models.FloatField(default=0.0)
    end_time = models.FloatField(default=0.0)
    metadata = models.TextField(default="{}")
    frame_count = models.IntegerField(default=0)
    start_index = models.IntegerField(default=0)
    start_frame = models.ForeignKey(Frame,null=True,related_name="segment_start")
    end_frame = models.ForeignKey(Frame, null=True,related_name="segment_end")

    class Meta:
        unique_together = (("video", "segment_index"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.segment_index)

    def path(self, media_root=None):
        if media_root:
            return "{}/{}/segments/{}.mp4".format(media_root, self.video_id, self.segment_index)
        else:
            return "{}/{}/segments/{}.mp4".format(settings.MEDIA_ROOT, self.video_id, self.segment_index)

    def framelist_path(self, media_root=None):
        if media_root:
            return "{}/{}/segments/{}.txt".format(media_root, self.video_id, self.segment_index)
        else:
            return "{}/{}/segments/{}.txt".format(settings.MEDIA_ROOT, self.video_id, self.segment_index)


class Region(models.Model):
    """
    Any 2D region over an image.
    Detections & Transforms have an associated image data.
    """
    ANNOTATION = 'A'
    DETECTION = 'D'
    SEGMENTATION = 'S'
    TRANSFORM = 'T'
    POLYGON = 'P'
    REGION_TYPES = (
        (ANNOTATION, 'Annotation'),
        (DETECTION, 'Detection'),
        (POLYGON, 'Polygon'),
        (SEGMENTATION, 'Segmentation'),
        (TRANSFORM, 'Transform'),
    )
    region_type = models.CharField(max_length=1,choices=REGION_TYPES,db_index=True)
    video = models.ForeignKey(Video)
    user = models.ForeignKey(User,null=True)
    frame = models.ForeignKey(Frame,null=True)
    event = models.ForeignKey(TEvent, null=True)  # TEvent that created this region
    parent_frame_index = models.IntegerField(default=-1)
    parent_segment_index = models.IntegerField(default=-1,null=True)
    text = models.TextField(default="")
    metadata = JSONField(blank=True,null=True)
    full_frame = models.BooleanField(default=False)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    polygon_points = JSONField(blank=True,null=True)
    created = models.DateTimeField('date created', auto_now_add=True)
    object_name = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    materialized = models.BooleanField(default=False)
    png = models.BooleanField(default=False)

    def clean(self):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index
        if self.parent_segment_index == -1 or self.parent_segment_index is None:
            self.parent_segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index
        if self.parent_segment_index == -1 or self.parent_segment_index is None:
            self.parent_segment_index = self.frame.segment_index
        super(Region, self).save(*args, **kwargs)

    def path(self,media_root=None):
        if media_root:
            return "{}/{}/regions/{}.jpg".format(media_root, self.video_id, self.pk)
        else:
            return "{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.pk)

    def frame_path(self,media_root=None):
        if media_root:
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.parent_frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.parent_frame_index)


class QueryResults(models.Model):
    query = models.ForeignKey(DVAPQL)
    retrieval_event = models.ForeignKey(TEvent,null=True)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)


class LOPQCodes(models.Model):
    retriever = models.ForeignKey(Retriever)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    fine = ArrayField(models.IntegerField(), default=[])
    coarse = ArrayField(models.IntegerField(), default=[])
    coarse_text = models.TextField(default="") # check if postgres built in text search
    fine_text = models.TextField(default="") # check if postgres built in text search can be used
    searcher_index = models.IntegerField()

    class Meta:
        unique_together = ('searcher_index', 'retriever')
        index_together = [["retriever", "searcher_index"],] # Very important manually verify in Postgres


class IndexEntries(models.Model):
    video = models.ForeignKey(Video)
    features_file_name = models.CharField(max_length=100)
    entries_file_name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100)
    indexer = models.ForeignKey(Indexer, null=True)
    indexer_shasum = models.CharField(max_length=40)
    detection_name = models.CharField(max_length=100)
    count = models.IntegerField()
    approximate = models.BooleanField(default=False)
    contains_frames = models.BooleanField(default=False)
    contains_detections = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)
    source = models.ForeignKey(TEvent, null=True)

    class Meta:
        unique_together = ('video', 'features_file_name',)

    def __unicode__(self):
        return "{} in {} index by {}".format(self.detection_name, self.algorithm, self.video.name)

    def npy_path(self, media_root=None):
        if media_root:
            return "{}/{}/indexes/{}".format(media_root, self.video_id, self.features_file_name)
        else:
            return "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, self.video_id, self.features_file_name)


class Tube(models.Model):
    """
    A tube is a collection of sequential frames / regions that track a certain object
    or describe a specific scene
    """
    video = models.ForeignKey(Video,null=True)
    frame_level = models.BooleanField(default=False)
    start_frame_index = models.IntegerField()
    end_frame_index = models.IntegerField()
    start_frame = models.ForeignKey(Frame,null=True,related_name="start_frame")
    end_frame = models.ForeignKey(Frame,null=True,related_name="end_frame")
    start_region = models.ForeignKey(Region,null=True,related_name="start_region")
    end_region = models.ForeignKey(Region,null=True,related_name="end_region")
    text = models.TextField(default="")
    metadata = JSONField(blank=True,null=True)
    source = models.ForeignKey(TEvent,null=True)


class Label(models.Model):
    name = models.CharField(max_length=200)
    set = models.CharField(max_length=200,default="")
    metadata = JSONField(blank=True,null=True)
    text = models.TextField(null=True,blank=True)
    created = models.DateTimeField('date created', auto_now_add=True)

    class Meta:
        unique_together = (("name", "set"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.name, self.set)


class FrameLabel(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame_index = models.IntegerField(default=-1)
    segment_index = models.IntegerField(null=True)
    frame = models.ForeignKey(Frame)
    label = models.ForeignKey(Label)
    event = models.ForeignKey(TEvent,null=True)

    def clean(self):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(FrameLabel, self).save(*args, **kwargs)


class RegionLabel(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame = models.ForeignKey(Frame,null=True)
    frame_index = models.IntegerField(default=-1)
    segment_index = models.IntegerField(null=True)
    region = models.ForeignKey(Region)
    label = models.ForeignKey(Label)
    event = models.ForeignKey(TEvent,null=True)

    def clean(self):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(RegionLabel, self).save(*args, **kwargs)


class SegmentLabel(models.Model):
    video = models.ForeignKey(Video,null=True)
    segment_index = models.IntegerField(default=-1)
    segment = models.ForeignKey(Segment)
    label = models.ForeignKey(Label)
    event = models.ForeignKey(TEvent, null=True)

    def clean(self):
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.segment.segment_index

    def save(self, *args, **kwargs):
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.segment.segment_index
        super(SegmentLabel, self).save(*args, **kwargs)


class TubeLabel(models.Model):
    video = models.ForeignKey(Video,null=True)
    tube = models.ForeignKey(Tube)
    label = models.ForeignKey(Label)
    event = models.ForeignKey(TEvent, null=True)


class VideoLabel(models.Model):
    video = models.ForeignKey(Video)
    label = models.ForeignKey(Label)
    event = models.ForeignKey(TEvent, null=True)


class DeletedVideo(models.Model):
    name = models.CharField(max_length=500,default="")
    description = models.TextField(default="")
    uploader = models.ForeignKey(User,null=True,related_name="user_uploader")
    url = models.TextField(default="")
    deleter = models.ForeignKey(User,related_name="user_deleter",null=True)
    original_pk = models.IntegerField()

    def __unicode__(self):
        return u'Deleted {}'.format(self.name)


class ManagementAction(models.Model):
    parent_task = models.CharField(max_length=500, default="")
    op = models.CharField(max_length=500, default="")
    host = models.CharField(max_length=500, default="")
    message = models.TextField()
    created = models.DateTimeField('date created', auto_now_add=True)


class StoredDVAPQL(models.Model):
    """
    Stored processes
    """
    SCHEDULE = 'S'
    PROCESS = 'V'
    QUERY = 'Q'
    TYPE_CHOICES = ((SCHEDULE, 'Schedule'), (PROCESS, 'Process'), (QUERY, 'Query'))
    process_type = models.CharField(max_length=1, choices=TYPE_CHOICES, default=QUERY,db_index=True)
    created = models.DateTimeField('date created', auto_now_add=True)
    creator = models.ForeignKey(User, null=True, related_name="script_creator")
    name = models.CharField(max_length=300,default="")
    description = models.TextField(blank=True,default="")
    script = JSONField(blank=True, null=True)


class SystemState(models.Model):
    created = models.DateTimeField('date created', auto_now_add=True)
    tasks = models.IntegerField(default=0)
    pending_tasks = models.IntegerField(default=0)
    completed_tasks = models.IntegerField(default=0)
    processes = models.IntegerField(default=0)
    pending_processes = models.IntegerField(default=0)
    completed_processes = models.IntegerField(default=0)


