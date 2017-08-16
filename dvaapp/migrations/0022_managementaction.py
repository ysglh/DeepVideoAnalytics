# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-08-16 16:27
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dvaapp', '0021_auto_20170816_0229'),
    ]

    operations = [
        migrations.CreateModel(
            name='ManagementAction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('parent_task', models.CharField(default='', max_length=500)),
                ('op', models.CharField(default='', max_length=500)),
                ('host', models.CharField(default='', max_length=500)),
                ('message', models.TextField()),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='date created')),
            ],
        ),
    ]
