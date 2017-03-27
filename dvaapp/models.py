from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User


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

    def __unicode__(self):
        return u'{}:{}'.format(self.video_id, self.frame_index)


class Detection(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame = models.ForeignKey(Frame)
    object_name = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    metadata = models.TextField(default="")


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


class ExternalDataset(models.Model):
    algorithm = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    count_images = models.IntegerField()
    count_entries = models.IntegerField()
    cached_image_count = models.IntegerField()
    index_downloaded = models.BooleanField(default=False)
    created = models.DateTimeField('date created', auto_now_add=True)


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
    label_name = models.CharField(max_length=200,unique=True)
    created = models.DateTimeField('date created', auto_now_add=True)


class Annotation(models.Model):
    """
    frame = models.ForeignKey(Frame)
    video = models.ForeignKey(Video)
    annotation = models.ForeignKey(Annotation,null=True)
    label =
    source = models.TextField()
    """
    video = models.ForeignKey(Video)
    user = models.ForeignKey(User,null=True)
    frame = models.ForeignKey(Frame,null=True)
    detection = models.ForeignKey(Detection,null=True)
    multi_frame = models.BooleanField(default=False)
    start_frame = models.ForeignKey(Frame,null=True,related_name='start_frame')
    end_frame = models.ForeignKey(Frame,null=True,related_name='end_frame')
    metadata_text = models.TextField(default="")
    name = models.TextField(default="unnamed_annotation")
    label_parent = models.ForeignKey(VLabel, null=True)
    label = models.TextField(default="empty")
    source = models.TextField(default="user_interface")
    full_frame = models.BooleanField(default=True)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    created = models.DateTimeField('date created', auto_now_add=True)

