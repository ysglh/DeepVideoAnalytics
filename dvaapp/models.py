from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField


class VDNServer(models.Model):
    url = models.URLField()
    name = models.CharField(max_length=200)
    last_response_datasets = models.TextField(default='[]')
    last_token = models.CharField(max_length=300, default="")


class VDNDataset(models.Model):
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


class Query(models.Model):
    created = models.DateTimeField('date created', auto_now_add=True)
    count = models.IntegerField(default=20) # retrieve 20 results per algorithm
    selected_indexers = ArrayField(models.CharField(max_length=30),default=[])
    excluded_index_entries_pk = ArrayField(models.IntegerField(), default=[])
    results = models.BooleanField(default=False)
    results_metadata = models.TextField(default="")
    user = models.ForeignKey(User, null=True)


class Video(models.Model):
    name = models.CharField(max_length=100,default="")
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
    detections = models.IntegerField(default=0)
    url = models.TextField(default="")
    youtube_video = models.BooleanField(default=False)
    query = models.BooleanField(default=False)
    parent_query = models.ForeignKey(Query,null=True)
    vdn_dataset = models.ForeignKey(VDNDataset, null=True)

    def __unicode__(self):
        return u'{}'.format(self.name)


class Frame(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame_index = models.IntegerField()
    name = models.CharField(max_length=200,null=True)
    subdir = models.TextField(default="") # Retains information if the source is a dataset for labeling

    class Meta:
        unique_together = (("video", "frame_index"),)

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.frame_index)


class Detection(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame = models.ForeignKey(Frame)
    parent_frame_index = models.IntegerField(default=-1)
    object_name = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    metadata = models.TextField(default="")
    vdn_dataset = models.ForeignKey(VDNDataset, null=True)
    vdn_key = models.IntegerField(default=-1)

    def clean(self):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index

    def save(self, *args, **kwargs):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index
        super(Detection, self).save(*args, **kwargs)


class IndexEntries(models.Model):
    video = models.ForeignKey(Video)
    features_file_name = models.CharField(max_length=100)
    entries_file_name = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100)
    detection_name = models.CharField(max_length=100)
    count = models.IntegerField()
    approximate = models.BooleanField(default=False)
    contains_frames = models.BooleanField(default=False)
    contains_detections = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)

    class Meta:
        unique_together = ('video', 'features_file_name',)

    def __unicode__(self):
        return "{} in {} index by {}".format(self.detection_name,self.algorithm,self.video.name)

class TEvent(models.Model):
    started = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)
    errored = models.BooleanField(default=False)
    error_message = models.TextField(default="")
    video = models.ForeignKey(Video,null=True)
    operation = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    seconds = models.FloatField(default=-1)


class QueryResults(models.Model):
    query = models.ForeignKey(Query)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Detection,null=True)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)


class VLabel(models.Model):
    UI = 'UI'
    DIRECTORY = 'DR'
    ALGO = 'AG'
    VDN = "VD"
    SOURCE_CHOICES = ((UI, 'User Interface'),(DIRECTORY, 'Directory Name'),(ALGO, 'Algorithm'),(VDN,"Visual Data Network"))
    label_name = models.CharField(max_length=200)
    source = models.CharField(max_length=2,choices=SOURCE_CHOICES,default=UI,)
    created = models.DateTimeField('date created', auto_now_add=True)
    vdn_dataset = models.ForeignKey(VDNDataset,null=True)
    video = models.ForeignKey(Video)
    class Meta:
        unique_together = ('source', 'label_name','video')


class Annotation(models.Model):
    video = models.ForeignKey(Video)
    user = models.ForeignKey(User,null=True)
    frame = models.ForeignKey(Frame,null=True)
    detection = models.ForeignKey(Detection,null=True)
    parent_frame_index = models.IntegerField(default=-1)
    metadata_text = models.TextField(default="")
    label_parent = models.ForeignKey(VLabel, null=True)
    label = models.TextField(default="")
    full_frame = models.BooleanField(default=True)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    created = models.DateTimeField('date created', auto_now_add=True)
    vdn_dataset = models.ForeignKey(VDNDataset,null=True)
    vdn_key = models.IntegerField(default=-1)


    def clean(self):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index

    def save(self, *args, **kwargs):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index
        super(Annotation, self).save(*args, **kwargs)


class Export(models.Model):
    video = models.ForeignKey(Video)
    file_name = models.CharField(max_length=200)
    started = models.DateTimeField('date created', auto_now_add=True)
    completed = models.BooleanField(default=False)


class S3Export(models.Model):
    video = models.ForeignKey(Video)
    key = models.CharField(max_length=300)
    bucket = models.CharField(max_length=300)
    region = models.CharField(max_length=300)
    started = models.DateTimeField('date created', auto_now_add=True)
    completed = models.BooleanField(default=False)


class S3Import(models.Model):
    video = models.ForeignKey(Video)
    requester_pays = models.BooleanField(default=False)
    key = models.CharField(max_length=300)
    bucket = models.CharField(max_length=300)
    region = models.CharField(max_length=300)
    started = models.DateTimeField('date created', auto_now_add=True)
    completed = models.BooleanField(default=False)


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


class ClusterCodes(models.Model):
    clusters = models.ForeignKey(Clusters)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    detection = models.ForeignKey(Detection,null=True)
    fine = ArrayField(models.IntegerField(), default=[])
    coarse = ArrayField(models.IntegerField(), default=[])
    coarse_text = models.TextField(default="") # check if postgres built in text search
    fine_text = models.TextField(default="") # check if postgres built in text search can be used
