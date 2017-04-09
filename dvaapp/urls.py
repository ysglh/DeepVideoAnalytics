from django.conf.urls import url,include
import views

from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'videos', views.VideoViewSet)
router.register(r'frames', views.FrameViewSet)
router.register(r'vlabels', views.VLabelViewSet)
router.register(r'annotations', views.AnnotationViewSet)
router.register(r'detections', views.DetectionViewSet)
router.register(r'queries', views.QueryViewSet)
router.register(r'queryresults', views.QueryResultsViewSet)
router.register(r'indexentries', views.IndexEntriesViewSet)
router.register(r'taskevents', views.TEventViewSet)
router.register(r'vdnservers', views.VDNServerViewSet)
router.register(r'vdndatasets', views.VDNDatasetViewSet)


urlpatterns = [
    url(r'^$', views.index, name='app'),
    url(r'^status$', views.status, name='status'),
    url(r'^tasks', views.tasks, name='tasks'),
    url(r'^indexes', views.indexes, name='indexes'),
    url(r'^annotations', views.annotations, name='annotations'),
    url(r'^detections', views.detections, name='detections'),
    url(r'^external', views.external, name='external'),
    url(r'^youtube$', views.yt, name='youtube'),
    url(r'^export_video', views.export_video, name='export_video'),
    url(r'^import_dataset', views.import_dataset, name='import_dataset'),
    url(r'^create_labels', views.create_label, name='create_labels'),
    url(r'^delete_labels', views.delete_label, name='delete_labels'),
    url(r'^videos/$', views.VideoList.as_view(),name="video_list"),
    url(r'^queries/$', views.QueryList.as_view()),
    url(r'^Search$', views.search),
    url(r'^videos/(?P<pk>\d+)/$', views.VideoDetail.as_view(), name='video_detail'),
    url(r'^frames/$', views.FrameList.as_view()),
    url(r'^frames/(?P<pk>\d+)/$', views.FrameDetail.as_view(), name='frames_detail'),
    url(r'^queries/(?P<pk>\d+)/$', views.QueryDetail.as_view(), name='query_detail'),
    url(r'^retry/(?P<pk>\d+)/$', views.retry_task, name='restart_task'),
    url(r'^requery/(?P<query_pk>\d+)/$', views.index, name='requery'),
    url(r'^query_frame/(?P<frame_pk>\d+)/$', views.index, name='query_frame'),
    url(r'^query_detection/(?P<detection_pk>\d+)/$', views.index, name='query_detection'),
    url(r'^annotate_frame/(?P<frame_pk>\d+)/$', views.annotate, name='annotate_frame'),
    url(r'^annotate_detection/(?P<detection_pk>\d+)/$', views.annotate, name='annotate_detection'),
    url(r'^delete', views.delete_object, name='delete_object'),
    url(r'^api/', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
