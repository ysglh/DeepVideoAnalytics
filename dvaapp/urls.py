from django.conf.urls import url, include
import views
from django.conf import settings
from django.contrib.auth import views as auth_views
from rest_framework import routers
import sys

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'videos', views.VideoViewSet)
router.register(r'tubes', views.TubeViewSet)
router.register(r'frames', views.FrameViewSet)
router.register(r'segments', views.SegmentViewSet)
router.register(r'regions', views.RegionViewSet)
router.register(r'queries', views.DVAPQLViewSet)
router.register(r'queryresults', views.QueryResultsViewSet)
router.register(r'indexentries', views.IndexEntriesViewSet)
router.register(r'events', views.TEventViewSet)
router.register(r'vdnservers', views.VDNServerViewSet)
router.register(r'vdndatasets', views.VDNDatasetViewSet)
router.register(r'clusters', views.ClustersViewSet)
router.register(r'clustercodes', views.ClusterCodesViewSet)


urlpatterns = [
    url(r'^app$', views.index, name='app'),
    url(r'^status$', views.status, name='status'),
    url(r'^tasks/$', views.TEventList.as_view(), name='tasks'),
    url(r'^video_tasks/(?P<pk>\d+)/$', views.TEventList.as_view(), name='video_tasks'),
    url(r'^video_tasks/(?P<pk>\d+)/(?P<status>\w+)/$', views.TEventList.as_view(), name='video_tasks_status'),
    url(r'^tasks/(?P<status>\w+)/$', views.TEventList.as_view(), name='tasks_filter'),
    url(r'^indexes', views.indexes, name='indexes'),
    url(r'^detectors/$', views.detectors, name='detectors'),
    url(r'^training/$', views.training, name='training'),
    url(r'^ocr/$', views.ocr, name='ocr'),
    url(r'^detectors/(?P<pk>\d+)/$', views.DetectionDetail.as_view(), name='detections_detail'),
    url(r'^textsearch', views.textsearch, name='textsearch'),
    url(r'^clustering/$', views.clustering, name='clustering'),
    url(r'^external', views.external, name='external'),
    url(r'^youtube$', views.yt, name='youtube'),
    url(r'^process/$', views.ProcessList.as_view(), name='process_list'),
    url(r'^process/(?P<pk>\d+)/$', views.ProcessDetail.as_view(), name='process_detail'),
    url(r'^export_video', views.export_video, name='export_video'),
    url(r'^delete_video', views.delete_video, name='delete_video'),
    url(r'^rename_video', views.rename_video, name='rename_video'),
    url(r'^import_dataset', views.import_dataset, name='import_dataset'),
    url(r'^import_detector', views.import_detector, name='import_detector'),
    url(r'^import_s3', views.import_s3, name='import_s3'),
    url(r'^submit_process', views.submit_process, name='submit_process'),
    url(r'^assign_video_labels', views.assign_video_labels, name='assign_video_labels'),
    # url(r'^delete_labels', views.delete_label, name='delete_labels'),
    url(r'^videos/$', views.VideoList.as_view(), name="video_list"),
    url(r'^queries/$', views.VisualSearchList.as_view()),
    url(r'^Search$', views.search),
    url(r'^videos/(?P<pk>\d+)/$', views.VideoDetail.as_view(), name='video_detail'),
    url(r'^clustering/(?P<pk>\d+)/$', views.ClustersDetails.as_view(), name='clusters_detail'),
    url(r'^vdn_datasets/(?P<pk>\d+)/$', views.VDNDatasetDetail.as_view(), name='vdn_dataset_detail'),
    url(r'^frames/$', views.FrameList.as_view()),
    url(r'^frames/(?P<pk>\d+)/$', views.FrameDetail.as_view(), name='frame_detail'),
    url(r'^segments/(?P<pk>\d+)/$', views.SegmentDetail.as_view(), name='segment_detail'),
    url(r'^queries/(?P<pk>\d+)/$', views.VisualSearchDetail.as_view(), name='query_detail'),
    url(r'^retry/$', views.retry_task, name='restart_task'),
    url(r'^failed/$', views.mark_task_failed, name='mark_task_failed'),
    url(r'^push/(?P<video_id>\d+)/$', views.push, name='push'),
    url(r'^coarse_code/(?P<pk>\d+)/(?P<coarse_code>\w+)$', views.coarse_code_detail, name='coarse_code_detail'),
    url(r'^segments/by_index/(?P<video_pk>\d+)/(?P<segment_index>\w+)$', views.segment_by_index, name='segment_by_index'),
    url(r'^requery/(?P<query_pk>\d+)/$', views.index, name='requery'),
    url(r'^query_frame/(?P<frame_pk>\d+)/$', views.index, name='query_frame'),
    url(r'^query_detection/(?P<detection_pk>\d+)/$', views.index, name='query_detection'),
    url(r'^annotate_frame/(?P<frame_pk>\d+)/$', views.annotate, name='annotate_frame'),
    url(r'^annotate_entire_frame/(?P<frame_pk>\d+)/$', views.annotate_entire_frame, name='annotate_entire_frame'),
    url(r'^delete', views.delete_object, name='delete_object'),
    url(r'^api/', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^login/$', auth_views.login, name='login'),
    url(r'^logout/$', auth_views.logout, name='logout'),
    url(r'^accounts/login/$', auth_views.login, name='login'),
    url(r'^accounts/logout/$', auth_views.logout, name='logout'),
    url(r'^password_reset/$', auth_views.password_reset, name='password_reset'),
    url(r'^accounts/profile/$', views.index, name='profile'),
]

if settings.DVA_PRIVATE_ENABLE and sys.platform != 'darwin':
    urlpatterns.append(url(r'^$', views.home, name='home'))
else:
    urlpatterns.append(url(r'^$', views.index, name='app_home'))
