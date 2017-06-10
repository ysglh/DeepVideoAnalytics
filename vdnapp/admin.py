from django.contrib import admin
from .models import Dataset, Annotation, Organization, Detector, Indexer
from rest_framework.authtoken.admin import TokenAdmin


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    pass


@admin.register(Detector)
class DetectorAdmin(admin.ModelAdmin):
    pass


@admin.register(Indexer)
class IndexerAdmin(admin.ModelAdmin):
    pass


@admin.register(Annotation)
class AnnotationsAdmin(admin.ModelAdmin):
    pass


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    pass

TokenAdmin.raw_id_fields = ('user',)
