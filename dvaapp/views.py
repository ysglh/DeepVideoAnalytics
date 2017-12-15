from django.conf import settings
import json
from .models import Video, Frame, DVAPQL, QueryResults, TEvent, IndexEntries, Region, \
    LOPQCodes, Tube,  Segment, FrameLabel, SegmentLabel, \
    VideoLabel, RegionLabel, TubeLabel, Label, \
    Retriever, SystemState, QueryRegion, QueryRegionResults, DeepModel, Worker
import serializers
from rest_framework import viewsets, mixins
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticatedOrReadOnly, IsAuthenticated
from .processing import DVAPQLProcess
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
import logging

try:
    from django.contrib.postgres.search import SearchVector
except ImportError:
    SearchVector = None
    logging.warning("Could not load Postgres full text search")


def user_check(user):
    return user.is_authenticated or settings.AUTH_DISABLED


def force_user_check(user):
    return user.is_authenticated


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = User.objects.all()
    serializer_class = serializers.UserSerializer


class SystemStateViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = SystemState.objects.all()
    serializer_class = serializers.SystemStateSerializer


class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Video.objects.all()
    serializer_class = serializers.VideoSerializer


class RetrieverViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Retriever.objects.all()
    serializer_class = serializers.RetrieverSerializer


class DeepModelViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = DeepModel.objects.all()
    serializer_class = serializers.DeepModelSerializer


class FrameViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Frame.objects.all()
    serializer_class = serializers.FrameSerializer
    filter_fields = ('frame_index', 'subdir', 'name', 'video')


class FrameLabelViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                        mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = FrameLabel.objects.all()
    serializer_class = serializers.FrameLabelSerializer
    filter_fields = ('frame_index','segment_index', 'video')


class RegionLabelViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                         mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = RegionLabel.objects.all()
    serializer_class = serializers.RegionLabelSerializer


class VideoLabelViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                        mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = VideoLabel.objects.all()
    serializer_class = serializers.VideoLabelSerializer


class SegmentLabelViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                          mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = SegmentLabel.objects.all()
    serializer_class = serializers.SegmentLabelSerializer


class TubeLabelViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin,
                       mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = TubeLabel.objects.all()
    serializer_class = serializers.TubeLabelSerializer


class SegmentViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Segment.objects.all()
    serializer_class = serializers.SegmentSerializer
    filter_fields = ('segment_index','video')


class QueryRegionViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = QueryRegion.objects.all()
    serializer_class = serializers.QueryRegionSerializer


class RegionViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin, mixins.DestroyModelMixin,
                    viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Region.objects.all()
    serializer_class = serializers.RegionSerializer
    filter_fields = ('video',)


class DVAPQLViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = DVAPQL.objects.all()
    serializer_class = serializers.DVAPQLSerializer

    def perform_create(self, serializer):
        instance = serializer.save(user=self.request.user)
        p = DVAPQLProcess(instance)
        spec = json.loads(self.request.POST.get('script'))
        p.create_from_json(spec, self.request.user)
        p.launch()

    def perform_update(self, serializer):
        """
        Immutable Not allowed
        :param serializer:
        :return:
        """
        raise NotImplementedError

    def perform_destroy(self, instance):
        """
        :param instance:
        :return:
        """
        raise ValueError, "Not allowed to delete"


class QueryResultsViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = QueryResults.objects.all()
    serializer_class = serializers.QueryResultsSerializer
    filter_fields = ('query',)


class QueryRegionResultsViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = QueryRegionResults.objects.all()
    serializer_class = serializers.QueryRegionResultsSerializer
    filter_fields = ('query',)


class TEventViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = TEvent.objects.all()
    serializer_class = serializers.TEventSerializer
    filter_fields = ('video', 'operation', 'completed', 'started', 'errored', 'parent_process')


class WorkerViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Worker.objects.all()
    serializer_class = serializers.WorkerSerializer


class IndexEntriesViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = IndexEntries.objects.all()
    serializer_class = serializers.IndexEntriesSerializer
    filter_fields = ('video', 'algorithm', 'detection_name')


class LabelViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Label.objects.all()
    serializer_class = serializers.LabelSerializer


class TubeViewSet(mixins.ListModelMixin, mixins.RetrieveModelMixin, mixins.CreateModelMixin, mixins.DestroyModelMixin, viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Tube.objects.all()
    serializer_class = serializers.TubeSerializer


class LOPQCodesViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = LOPQCodes.objects.all()
    serializer_class = serializers.LOPQCodesSerializer
