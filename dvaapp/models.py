from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User


class VDNObject(models.Model):
    url = models.URLField()

class Query(models.Model):
    created = models.DateTimeField('date created', auto_now_add=True)
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
    vdn_key = models.IntegerField(default=-1)
    vdn_source = models.URLField(default="")
    metadata = models.TextField(default="")

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
    vdn_key = models.IntegerField(default=-1)
    vdn_source = models.URLField(default="")

    class Meta:
        unique_together = ('video', 'features_file_name',)


class TEvent(models.Model):
    started = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)
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
    SOURCE_CHOICES = ((UI, 'User Interface'),(DIRECTORY, 'Directory Name'),(ALGO, 'Algorithm'))
    label_name = models.CharField(max_length=200)
    source = models.CharField(max_length=2,choices=SOURCE_CHOICES,default=UI,)
    created = models.DateTimeField('date created', auto_now_add=True)
    class Meta:
        unique_together = ('source', 'label_name',)


class Annotation(models.Model):
    video = models.ForeignKey(Video)
    user = models.ForeignKey(User,null=True)
    frame = models.ForeignKey(Frame,null=True)
    detection = models.ForeignKey(Detection,null=True)
    parent_frame_index = models.IntegerField(default=-1)
    metadata_text = models.TextField(default="")
    label_parent = models.ForeignKey(VLabel, null=True)
    label = models.TextField(default="empty")
    full_frame = models.BooleanField(default=True)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    vdn_key = models.IntegerField(default=-1)
    vdn_source = models.URLField(default="")
    created = models.DateTimeField('date created', auto_now_add=True)

    def clean(self):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index

    def save(self, *args, **kwargs):
        if self.parent_frame_index == -1 or self.parent_frame_index is None:
            self.parent_frame_index = self.frame.frame_index
        super(Annotation, self).save(*args, **kwargs)


class VDNServer(models.Model):
    url = models.URLField()
    name = models.CharField(max_length=200)
    last_response_datasets = models.TextField(default='[]')

class VDNDataset(models.Model):
    server = models.ForeignKey(VDNServer)
    child_video = models.ForeignKey(Video,null=True)
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

class Export(models.Model):
    video = models.ForeignKey(Video)
    file_name = models.CharField(max_length=200)
    started = models.DateTimeField('date created', auto_now_add=True)
    completed = models.BooleanField(default=False)