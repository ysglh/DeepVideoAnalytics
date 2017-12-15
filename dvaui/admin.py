# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin
from .models import StoredDVAPQL


@admin.register(StoredDVAPQL)
class StoredDVAPQLAdmin(admin.ModelAdmin):
    pass
