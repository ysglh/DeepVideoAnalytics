from django.contrib import admin
from .models import Video,Frame,TEvent,IndexEntries,QueryResults,Query,VLabel,VDNServer,VDNDataset, ClusterCodes, Clusters, Region


@admin.register(VLabel)
class AnnotationTagAdmin(admin.ModelAdmin):
    pass


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    pass


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryResults)
class QueryResultsAdmin(admin.ModelAdmin):
    pass


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    pass

@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    pass


@admin.register(IndexEntries)
class IndexEntriesAdmin(admin.ModelAdmin):
    pass


@admin.register(VDNServer)
class VDNServerAdmin(admin.ModelAdmin):
    pass


@admin.register(VDNDataset)
class VDNDatasetAdmin(admin.ModelAdmin):
    pass


@admin.register(TEvent)
class TEventAdmin(admin.ModelAdmin):
    pass



@admin.register(Clusters)
class ClustersAdmin(admin.ModelAdmin):
    pass


@admin.register(ClusterCodes)
class ClusterCodesAdmin(admin.ModelAdmin):
    pass



