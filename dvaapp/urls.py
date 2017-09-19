import views
from rest_framework import routers
from django.conf.urls import url,include

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'videos', views.VideoViewSet)
router.register(r'analyzers', views.AnalyzerViewSet)
router.register(r'detectors', views.DetectorViewSet)
router.register(r'indexers', views.IndexerViewSet)
router.register(r'retrievers', views.RetrieverViewSet)
router.register(r'tubes', views.TubeViewSet)
router.register(r'frames', views.FrameViewSet)
router.register(r'framelabels', views.FrameLabelViewSet)
router.register(r'regionlabels', views.RegionLabelViewSet)
router.register(r'segmentlabels', views.SegmentLabelViewSet)
router.register(r'tubelabels', views.TubeLabelViewSet)
router.register(r'videolabels', views.VideoLabelViewSet)
router.register(r'labels', views.LabelViewSet)
router.register(r'segments', views.SegmentViewSet)
router.register(r'regions', views.RegionViewSet)
router.register(r'queries', views.DVAPQLViewSet)
router.register(r'queryresults', views.QueryResultsViewSet)
router.register(r'queryregionresults', views.QueryRegionResultsViewSet)
router.register(r'queryregions', views.QueryRegionViewSet)
router.register(r'indexentries', views.IndexEntriesViewSet)
router.register(r'events', views.TEventViewSet)
router.register(r'vdnservers', views.VDNServerViewSet)
router.register(r'lopqcodes', views.LOPQCodesViewSet)
router.register(r'system_state', views.SystemStateViewSet)

urlpatterns = [url(r'', include(router.urls)),]