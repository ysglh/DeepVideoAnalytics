from __future__ import unicode_literals
import os, json, gzip, sys, shutil, zipfile
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField
from django.conf import settings
from . import fs
try:
    import numpy as np
except ImportError:
    pass


class Worker(models.Model):
    queue_name = models.CharField(max_length=500, default="")
    host = models.CharField(max_length=500, default="")
    pid = models.IntegerField()
    alive = models.BooleanField(default=True)
    created = models.DateTimeField('date created', auto_now_add=True)


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
    completed = models.BooleanField(default=False)


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
        if not (media_root is None):
            return "{}/{}/video/{}.mp4".format(media_root, self.pk, self.pk)
        else:
            return "{}/{}/video/{}.mp4".format(settings.MEDIA_ROOT,self.pk,self.pk)

    def get_frame_list(self,media_root=None):
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        framelist_path = "{}/{}/framelist".format(media_root, self.pk)
        if os.path.isfile('{}.json'.format(framelist_path)):
            return json.load(file('{}.json'.format(framelist_path)))
        elif os.path.isfile('{}.gz'.format(framelist_path)):
            return json.load(gzip.GzipFile('{}.gz'.format(framelist_path)))
        else:
            raise ValueError("Frame list could not be found at {}".format(framelist_path))

    def create_directory(self, create_subdirs=True):
        d = '{}/{}'.format(settings.MEDIA_ROOT, self.pk)
        if not os.path.exists(d):
            try:
                os.mkdir(d)
            except OSError:
                pass
        if create_subdirs:
            for s in ['video','frames','segments','indexes','regions','transforms','audio']:
                d = '{}/{}/{}/'.format(settings.MEDIA_ROOT, self.pk, s)
                if not os.path.exists(d):
                    try:
                        os.mkdir(d)
                    except OSError:
                        pass


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
    worker = models.ForeignKey(Worker, null=True)
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


class TrainedModel(models.Model):
    """
    A model Model
    """
    TENSORFLOW = 'T'
    CAFFE = 'C'
    PYTORCH = 'P'
    OPENCV = 'O'
    MXNET = 'M'
    MODES = (
        (TENSORFLOW, 'Tensorflow'),
        (CAFFE, 'Caffe'),
        (PYTORCH, 'Pytorch'),
        (OPENCV, 'OpenCV'),
        (MXNET, 'MXNet'),
    )
    INDEXER = 'I'
    DETECTOR = 'D'
    ANALYZER = 'A'
    SEGMENTER = 'S'
    MTYPE = (
        (INDEXER, 'Indexer'),
        (DETECTOR, 'Detector'),
        (ANALYZER, 'Analyzer'),
        (SEGMENTER, 'Segmenter'),
    )
    YOLO = "Y"
    TFD = "T"
    DETECTOR_TYPES = (
        (TFD, 'Tensorflow'),
        (YOLO, 'YOLO V2'),
    )
    detector_type = models.CharField(max_length=1,choices=DETECTOR_TYPES,db_index=True,null=True)
    mode = models.CharField(max_length=1,choices=MODES,db_index=True,default=TENSORFLOW)
    model_type = models.CharField(max_length=1,choices=MTYPE,db_index=True,default=INDEXER)
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    shasum = models.CharField(max_length=40,null=True,unique=True)
    model_filename = models.CharField(max_length=200,default="",null=True)
    input_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_size = models.CharField(max_length=300,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    arguments = models.TextField(default="")
    phase_1_log = models.TextField(default="")
    phase_2_log = models.TextField(default="")
    class_distribution = models.TextField(default="")
    class_names = models.TextField(default="")
    frames_count = models.IntegerField(default=0)
    boxes_count = models.IntegerField(default=0)
    source = models.ForeignKey(TEvent, null=True)
    trained = models.BooleanField(default=False)
    class_index_to_string = JSONField(null=True,blank=True)
    url = models.CharField(max_length=200,default="")
    files = JSONField(null=True,blank=True)
    produces_labels = models.BooleanField(default=False)
    produces_json = models.BooleanField(default=False)
    produces_text = models.BooleanField(default=False)
    # Following allows us to have a hierarchy of models (E.g. inception pretrained -> inception fine tuned)
    parent = models.ForeignKey('self', null=True)

    def create_directory(self,create_subdirs=True):
        try:
            os.mkdir('{}/models/{}'.format(settings.MEDIA_ROOT, self.pk))
        except:
            pass

    def get_model_path(self,root_dir=None):
        if root_dir is None:
            root_dir = settings.MEDIA_ROOT
        if self.model_filename:
            return "{}/models/{}/{}".format(root_dir,self.pk,self.model_filename)
        elif self.files:
            return "{}/models/{}/{}".format(root_dir, self.pk, self.files[0]['filename'])
        else:
            return None

    def get_yolo_args(self):
        model_dir = "{}/models/{}/".format(settings.MEDIA_ROOT, self.pk)
        class_names = {k: v for k, v in json.loads(self.class_names)}
        args = {'root_dir': model_dir,
                'detector_pk': self.pk,
                'class_names':{i: k for k, i in class_names.items()}
                }
        return args

    def get_class_dist(self):
        return json.loads(self.class_distribution) if self.class_distribution.strip() else {}

    def download(self):
        root_dir = settings.MEDIA_ROOT
        model_type_dir = "{}/models/".format(root_dir)
        if not os.path.isdir(model_type_dir):
            os.mkdir(model_type_dir)
        model_dir = "{}/models/{}".format(root_dir, self.pk)
        if not os.path.isdir(model_dir):
            try:
                os.mkdir(model_dir)
            except:
                pass
        for m in self.files:
            dlpath = "{}/{}".format(model_dir,m['filename'])
            if m['url'].startswith('/'):
                shutil.copy(m['url'], dlpath)
            else:
                fs.get_path_to_file(m['url'],dlpath)
            if settings.DISABLE_NFS and sys.platform != 'darwin':
                fs.upload_file_to_remote("/models/{}/{}".format(self.pk,m['filename']))
        if self.model_type == TrainedModel.DETECTOR and self.detector_type == TrainedModel.YOLO:
            source_zip = "{}/models/{}/model.zip".format(settings.MEDIA_ROOT, self.pk)
            zipf = zipfile.ZipFile(source_zip, 'r')
            zipf.extractall("{}/models/{}/".format(settings.MEDIA_ROOT, self.pk))
            zipf.close()
            os.remove(source_zip)
            self.phase_1_log = file("{}/models/{}/phase_1.log".format(settings.MEDIA_ROOT, self.pk)).read()
            self.phase_2_log = file("{}/models/{}/phase_2.log".format(settings.MEDIA_ROOT, self.pk)).read()
            with open("{}/models/{}/input.json".format(settings.MEDIA_ROOT, self.pk)) as fh:
                metadata = json.load(fh)
            if 'class_distribution' in metadata:
                self.class_distribution = json.dumps(metadata['class_distribution'])
            else:
                self.class_distribution = json.dumps(metadata['class_names'])
                self.class_names = json.dumps(metadata['class_names'])
            self.save()

    def ensure(self):
        for m in self.files:
            dlpath = "{}/models/{}/{}".format(settings.MEDIA_ROOT, self.pk, m['filename'])
            if not os.path.isfile(dlpath):
                fs.ensure("/models/{}/{}".format(self.pk,m['filename']))


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


class QueryIndexVector(models.Model):
    event = models.OneToOneField(TEvent)
    vector = models.BinaryField()
    created = models.DateTimeField('date created', auto_now_add=True)


class Frame(models.Model):
    video = models.ForeignKey(Video)
    event = models.ForeignKey(TEvent,null=True)
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
        if not (media_root is None):
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,self.video_id,self.frame_index)

    def original_path(self):
        return self.name


class Segment(models.Model):
    """
    A video segment useful for parallel dense decoding+processing as well as streaming
    """
    video = models.ForeignKey(Video)
    segment_index = models.IntegerField()
    start_time = models.FloatField(default=0.0)
    end_time = models.FloatField(default=0.0)
    event = models.ForeignKey(TEvent,null=True)
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
        if not (media_root is None):
            return "{}/{}/segments/{}.mp4".format(media_root, self.video_id, self.segment_index)
        else:
            return "{}/{}/segments/{}.mp4".format(settings.MEDIA_ROOT, self.video_id, self.segment_index)

    def framelist_path(self, media_root=None):
        if not (media_root is None):
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
    frame_index = models.IntegerField(default=-1)
    segment_index = models.IntegerField(default=-1,null=True)
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
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index

    def save(self, *args, **kwargs):
        if self.frame_index == -1 or self.frame_index is None:
            self.frame_index = self.frame.frame_index
        if self.segment_index == -1 or self.segment_index is None:
            self.segment_index = self.frame.segment_index
        super(Region, self).save(*args, **kwargs)

    def path(self,media_root=None,temp_root=None):
        if temp_root:
            return "{}/{}_{}.jpg".format(temp_root, self.video_id, self.pk)
        elif not (media_root is None):
            return "{}/{}/regions/{}.jpg".format(media_root, self.video_id, self.pk)
        else:
            return "{}/{}/regions/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.pk)

    def frame_path(self,media_root=None):
        if not (media_root is None):
            return "{}/{}/frames/{}.jpg".format(media_root, self.video_id, self.frame_index)
        else:
            return "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT, self.video_id, self.frame_index)


class QueryRegion(models.Model):
    """
    Any 2D region over a query image.
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
    query = models.ForeignKey(DVAPQL)
    event = models.ForeignKey(TEvent, null=True)  # TEvent that created this region
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
    png = models.BooleanField(default=False)


class QueryResults(models.Model):
    query = models.ForeignKey(DVAPQL)
    retrieval_event = models.ForeignKey(TEvent,null=True)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)


class QueryRegionResults(models.Model):
    query = models.ForeignKey(DVAPQL)
    query_region = models.ForeignKey(QueryRegion)
    retrieval_event = models.ForeignKey(TEvent,null=True)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)


class IndexEntries(models.Model):
    video = models.ForeignKey(Video)
    features_file_name = models.CharField(max_length=100)
    entries_file_name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100)
    indexer = models.ForeignKey(TrainedModel, null=True)
    indexer_shasum = models.CharField(max_length=40)
    detection_name = models.CharField(max_length=100)
    count = models.IntegerField()
    approximate = models.BooleanField(default=False)
    contains_frames = models.BooleanField(default=False)
    contains_detections = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)
    event = models.ForeignKey(TEvent, null=True)

    class Meta:
        unique_together = ('video', 'features_file_name',)

    def __unicode__(self):
        return "{} in {} index by {}".format(self.detection_name, self.algorithm, self.video.name)

    def npy_path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/indexes/{}".format(media_root, self.video_id, self.features_file_name)
        else:
            return "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, self.video_id, self.features_file_name)

    def entries_path(self, media_root=None):
        if not (media_root is None):
            return "{}/{}/indexes/{}".format(media_root, self.video_id, self.entries_file_name)
        else:
            return "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, self.video_id, self.entries_file_name)

    def load_index(self,media_root=None):
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        video_dir = "{}/{}".format(media_root, self.video_id)
        if not os.path.isdir(video_dir):
            os.mkdir(video_dir)
        index_dir = "{}/{}/indexes".format(media_root, self.video_id)
        if not os.path.isdir(index_dir):
            os.mkdir(index_dir)
        dirnames = {}
        fs.ensure(self.entries_path(media_root=''),dirnames,media_root)
        fs.ensure(self.npy_path(media_root=''),dirnames,media_root)
        vectors = np.load(self.npy_path(media_root))
        vector_entries = json.load(file(self.entries_path(media_root)))
        return vectors,vector_entries


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
    ping_index = models.IntegerField(null=True)


class SystemState(models.Model):
    created = models.DateTimeField('date created', auto_now_add=True)
    tasks = models.IntegerField(default=0)
    pending_tasks = models.IntegerField(default=0)
    completed_tasks = models.IntegerField(default=0)
    processes = models.IntegerField(default=0)
    pending_processes = models.IntegerField(default=0)
    completed_processes = models.IntegerField(default=0)
    queues = JSONField(blank=True,null=True)
    hosts = JSONField(blank=True,null=True)


class QueryRegionIndexVector(models.Model):
    event = models.ForeignKey(TEvent)
    query_region = models.ForeignKey(QueryRegion)
    vector = models.BinaryField()
    created = models.DateTimeField('date created', auto_now_add=True)


class TrainingSet(models.Model):
    DETECTION = 'D'
    INDEXING = 'I'
    LOPQINDEX = 'A'
    CLASSIFICATION = 'C'
    TRAIN_TASK_TYPES = (
        (DETECTION, 'Detection'),
        (INDEXING, 'Indexing'),
        (CLASSIFICATION, 'Classication')
    )
    IMAGES = 'I'
    VIDEOS = 'V'
    INSTANCE_TYPES = (
        (IMAGES, 'images'),
        (VIDEOS, 'videos'),
    )
    event = models.ForeignKey(TEvent)
    training_task_type = models.CharField(max_length=1,choices=TRAIN_TASK_TYPES,db_index=True,default=DETECTION)
    instance_type = models.CharField(max_length=1,choices=INSTANCE_TYPES,db_index=True,default=IMAGES)
    count = models.IntegerField(null=True)
    name = models.CharField(max_length=500,default="")
    created = models.DateTimeField('date created', auto_now_add=True)


class WorkerRequest(models.Model):
    """ TODO(future): This model can be stored in Redis rather than DB """
    queue_name = models.CharField(max_length=500, default="")
    created = models.DateTimeField('date created', auto_now_add=True)