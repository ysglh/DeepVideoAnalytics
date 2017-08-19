from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField


class VDNServer(models.Model):
    """
    A VDN server
    """
    url = models.URLField()
    name = models.CharField(max_length=200)
    last_response_datasets = models.TextField(default='[]')
    last_response_detectors = models.TextField(default='[]')
    last_token = models.CharField(max_length=300, default="")


class VDNDataset(models.Model):
    """
    A VDN dataset
    """
    server = models.ForeignKey(VDNServer)
    response = models.TextField(default="")
    date_imported = models.DateTimeField('date created', auto_now_add=True)
    name = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    download_url = models.TextField(default="")
    url = models.TextField(default="")
    aws_requester_pays = models.BooleanField(default=False)
    aws_region = models.TextField(default="")
    aws_bucket = models.TextField(default="")
    aws_key = models.TextField(default="")
    root = models.BooleanField(default=True)
    parent_local = models.ForeignKey('self',null=True)
    organization_url = models.TextField()


class VDNDetector(models.Model):
    """
    A VDN detector
    """
    server = models.ForeignKey(VDNServer)
    response = models.TextField(default="")
    date_imported = models.DateTimeField('date created', auto_now_add=True)
    name = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    download_url = models.TextField(default="")
    url = models.TextField(default="")
    aws_requester_pays = models.BooleanField(default=False)
    aws_region = models.TextField(default="")
    aws_bucket = models.TextField(default="")
    aws_key = models.TextField(default="")
    organization_url = models.TextField()


class CustomIndexer(models.Model):
    """
    A custom indexer that can be used with any TF (eventually pytorch) network
    """
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    model_filename = models.CharField(max_length=200,default="")
    # vdn_detector = models.ForeignKey(VDNDetector,null=True)
    input_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_name = models.CharField(max_length=300,default="")
    embedding_layer_size = models.CharField(max_length=300,default="")
    indexer_queue = models.CharField(max_length=300,default="")
    retriever_queue = models.CharField(max_length=300,default="")


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


class IndexerQuery(models.Model):
    parent_query = models.ForeignKey(DVAPQL)
    created = models.DateTimeField('date created', auto_now_add=True)
    count = models.IntegerField(default=20)
    algorithm = models.CharField(max_length=500,default="")
    indexer = models.ForeignKey(CustomIndexer,null=True)
    excluded_index_entries_pk = ArrayField(models.IntegerField(), default=[])
    vector = models.BinaryField(null=True)
    results = models.BooleanField(default=False)
    metadata = models.TextField(default="")
    source_filter_json = JSONField(blank=True,null=True)
    approximate = models.BooleanField(default=False)
    user = models.ForeignKey(User, null=True)


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
    query = models.BooleanField(default=False)
    parent_process = models.ForeignKey(DVAPQL,null=True)
    vdn_dataset = models.ForeignKey(VDNDataset, null=True)

    def __unicode__(self):
        return u'{}'.format(self.name)


class Clusters(models.Model):
    excluded_index_entries_pk = ArrayField(models.IntegerField(), default=[])
    included_index_entries_pk = ArrayField(models.IntegerField(), default=[])
    train_fraction = models.FloatField(default=0.8) # by default use 80% of data for training
    algorithm = models.CharField(max_length=50,default='LOPQ')    # LOPQ
    indexer_algorithm = models.CharField(max_length=50)
    cluster_count = models.IntegerField(default=0)
    pca_file_name = models.CharField(max_length=200,default="")
    model_file_name = models.CharField(max_length=200, default="")
    components = models.IntegerField(default=64) # computer 64 principal components
    started = models.DateTimeField('date created', auto_now_add=True)
    completed = models.BooleanField(default=False)
    m = models.IntegerField(default=16)
    v = models.IntegerField(default=16)
    sub = models.IntegerField(default=256)


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
    seconds = models.FloatField(default=-1)
    arguments = JSONField(blank=True,null=True)
    task_id = models.TextField(null=True)
    parent = models.ForeignKey('self',null=True)
    parent_process = models.ForeignKey(DVAPQL,null=True)


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
    polygon_points_json = models.TextField(default="[]")
    created = models.DateTimeField('date created', auto_now_add=True)
    vdn_dataset = models.ForeignKey(VDNDataset,null=True)
    vdn_key = models.IntegerField(default=-1)
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


class QueryResults(models.Model):
    query = models.ForeignKey(DVAPQL)
    indexerquery = models.ForeignKey(IndexerQuery)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)


class ClusterCodes(models.Model):
    clusters = models.ForeignKey(Clusters)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Region,null=True)
    fine = ArrayField(models.IntegerField(), default=[])
    coarse = ArrayField(models.IntegerField(), default=[])
    coarse_text = models.TextField(default="") # check if postgres built in text search
    fine_text = models.TextField(default="") # check if postgres built in text search can be used
    searcher_index = models.IntegerField()

    class Meta:
        unique_together = ('searcher_index', 'clusters')
        index_together = [["clusters", "searcher_index"],] # Very important manually verify in Postgres


class IndexEntries(models.Model):
    video = models.ForeignKey(Video)
    features_file_name = models.CharField(max_length=100)
    entries_file_name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100)
    indexer = models.ForeignKey(CustomIndexer, null=True)
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


class CustomDetector(models.Model):
    name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100,default="")
    model_filename = models.CharField(max_length=200,default="")
    vdn_detector = models.ForeignKey(VDNDetector,null=True)
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
