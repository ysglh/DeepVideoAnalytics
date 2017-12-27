# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import subprocess, os, json, logging, tempfile, shutil
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField


class ExternalServer(models.Model):
    name = models.CharField(max_length=300,default="",unique=True)
    url = models.CharField(max_length=1000,default="",unique=True)
    created = models.DateTimeField('date created', auto_now_add=True)

    class Meta:
        unique_together = (("name", "url"),)

    def pull(self):
        errors = []
        cwd = tempfile.mkdtemp()
        gitpull = subprocess.Popen(['git', 'clone', self.url, self.name], cwd=cwd)
        gitpull.wait()
        for root, directories, filenames in os.walk("{}/{}/".format(cwd,self.name)):
            for filename in filenames:
                if filename.endswith('.json'):
                    try:
                        j = json.load(file(os.path.join(root,filename)))
                    except:
                        errors.append(filename)
                    else:
                        url = self.url
                        if url.endswith('/'):
                            url = url[:-1]
                        relpath = os.path.join(root,filename)[len(cwd)+1+len(self.name):]
                        if relpath.startswith('/'):
                            relpath = relpath[1:]
                        flname = "{}/{}".format(url,relpath)
                        p, _ = StoredDVAPQL.objects.get_or_create(name=flname,server=self)
                        p.server = self
                        p.process_type = StoredDVAPQL.PROCESS
                        p.script = j
                        p.description = j.get('description',"")
                        p.save()
        shutil.rmtree(cwd)
        if errors:
            logging.warning("Could not import {}".format(errors))
        return errors


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
    server = models.ForeignKey(ExternalServer,null=True)
    script = JSONField(blank=True, null=True)


