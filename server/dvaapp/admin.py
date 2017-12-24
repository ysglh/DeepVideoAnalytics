from django.contrib import admin
from .models import Video, Frame, TEvent, IndexEntries, QueryResults, DVAPQL, \
    Region, Tube, Segment, DeletedVideo, \
    VideoLabel, FrameLabel, RegionLabel, TubeLabel, SegmentLabel, Label, ManagementAction, \
    TrainedModel, Retriever, SystemState, Worker, QueryRegion, QueryRegionIndexVector, \
    QueryRegionResults


@admin.register(SystemState)
class SystemStateAdmin(admin.ModelAdmin):
    pass


@admin.register(Worker)
class WorkerAdmin(admin.ModelAdmin):
    pass


@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    pass


@admin.register(VideoLabel)
class VideoLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(FrameLabel)
class FrameLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(SegmentLabel)
class SegmentLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(RegionLabel)
class RegionLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(TubeLabel)
class TubeLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(Segment)
class SegmentAdmin(admin.ModelAdmin):
    pass


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    pass


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    pass


@admin.register(DeletedVideo)
class DeletedVideoAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryResults)
class QueryResultsAdmin(admin.ModelAdmin):
    pass


@admin.register(DVAPQL)
class DVAPQLAdmin(admin.ModelAdmin):
    pass

@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    pass


@admin.register(IndexEntries)
class IndexEntriesAdmin(admin.ModelAdmin):
    pass


@admin.register(TEvent)
class TEventAdmin(admin.ModelAdmin):
    pass


@admin.register(Tube)
class TubeAdmin(admin.ModelAdmin):
    pass


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    pass


@admin.register(Retriever)
class RetrieverAdmin(admin.ModelAdmin):
    pass


@admin.register(ManagementAction)
class ManagementActionAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryRegion)
class QueryRegionAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryRegionIndexVector)
class QueryRegionIndexVectorAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryRegionResults)
class QueryRegionResultsAdmin(admin.ModelAdmin):
    pass
