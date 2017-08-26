from django.contrib import admin
from .models import VDNRemoteDataset, Organization, VDNRemoteDetector
from rest_framework.authtoken.admin import TokenAdmin


@admin.register(VDNRemoteDataset)
class VDNRemoteDatasetAdmin(admin.ModelAdmin):
    pass


@admin.register(VDNRemoteDetector)
class VDNRemoteDetectorAdmin(admin.ModelAdmin):
    pass




@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    pass

TokenAdmin.raw_id_fields = ('user',)
