# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField, JSONField


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
