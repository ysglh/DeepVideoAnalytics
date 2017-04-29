from __future__ import unicode_literals
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)


class Organization(models.Model):
    name = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    user = models.OneToOneField(User)


class Dataset(models.Model):
    name = models.CharField(max_length=100,default="")
    root = models.BooleanField(default=True)
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    organization = models.ForeignKey(Organization)
    parent_url = models.CharField(max_length=300,default="")
    download_url = models.TextField(default="",blank=True)
    aws_requester_pays = models.BooleanField(default=False)
    aws_region = models.TextField(default="",blank=True)
    aws_bucket = models.TextField(default="",blank=True)
    aws_key = models.TextField(default="",blank=True)

    def __unicode__(self):
        return u'{}'.format(self.name)


class Annotation(models.Model):
    dataset = models.ForeignKey(Dataset,null=True)
    parent_frame_index = models.IntegerField(default=-1)
    metadata_text = models.TextField(default="")
    label = models.TextField(default="empty")
    full_frame = models.BooleanField(default=True)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    created = models.DateTimeField('date created', auto_now_add=True)


class Indexer(models.Model):
    name = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    tfgraph_url = models.TextField(default="",blank=True)


class Detector(models.Model):
    name = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    tfgraph_url = models.TextField(default="",blank=True)