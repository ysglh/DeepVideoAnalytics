from django.shortcuts import render,redirect
from django.conf import settings
from django.http import JsonResponse
import requests
import json
from django.views.generic import ListView,DetailView
from .forms import UploadFileForm,YTVideoForm,AnnotationForm
from .models import Video,Frame,Query,QueryResults,TEvent,IndexEntries,VDNDataset, Region, VDNServer, ClusterCodes, Clusters, AppliedLabel, Scene
from dva.celery import app
import serializers
from rest_framework import viewsets,mixins
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticatedOrReadOnly,IsAuthenticated
from django.db.models import Count
from celery.exceptions import TimeoutError
import math
from django.db.models import Max
from shared import create_video_folders,handle_uploaded_file,create_annotation,create_child_vdn_dataset,\
    create_query,create_root_vdn_dataset,handle_youtube_video,pull_vdn_dataset_list,import_vdn_dataset_url
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.mixins import UserPassesTestMixin
import logging
import base64
import pickle


def user_check(user):
    return user.is_authenticated or settings.AUTH_DISABLED


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = User.objects.all()
    serializer_class = serializers.UserSerializer


class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Video.objects.all()
    serializer_class = serializers.VideoSerializer


class FrameViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,) 
    queryset = Frame.objects.all()
    serializer_class = serializers.FrameSerializer
    filter_fields = ('frame_index', 'subdir', 'name', 'video')


class RegionViewSet(mixins.ListModelMixin,mixins.RetrieveModelMixin,mixins.CreateModelMixin,mixins.DestroyModelMixin,viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Region.objects.all()
    serializer_class = serializers.RegionSerializer
    filter_fields = ('video', 'frame', 'object_name')


class QueryViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Query.objects.all()
    serializer_class = serializers.QuerySerializer


class QueryResultsViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = QueryResults.objects.all()
    serializer_class = serializers.QueryResultsSerializer
    filter_fields = ('frame', 'video')


class TEventViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = TEvent.objects.all()
    serializer_class = serializers.TEventSerializer
    filter_fields = ('video','operation')


class IndexEntriesViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = IndexEntries.objects.all()
    serializer_class = serializers.IndexEntriesSerializer
    filter_fields = ('video','algorithm','detection_name')


class AppliedLabelViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = AppliedLabel.objects.all()
    serializer_class = serializers.AppliedLabelSerializer


class VDNServerViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = VDNServer.objects.all()
    serializer_class = serializers.VDNServerSerializer


class VDNDatasetViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = VDNDataset.objects.all()
    serializer_class = serializers.VDNDatasetSerializer


class SceneViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Scene.objects.all()
    serializer_class = serializers.SceneSerializer


class ClustersViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = Clusters.objects.all()
    serializer_class = serializers.ClustersSerializer


class ClusterCodesViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,) if settings.AUTH_DISABLED else (IsAuthenticated,)
    queryset = ClusterCodes.objects.all()
    serializer_class = serializers.ClusterCodesSerializer


class VideoList(UserPassesTestMixin,ListView):
    model = Video
    paginate_by = 100

    def get_context_data(self, **kwargs):
        context = super(VideoList, self).get_context_data(**kwargs)
        context['exports'] = TEvent.objects.all().filter(event_type=TEvent.EXPORT)
        context['s3_exports'] = TEvent.objects.all().filter(event_type=TEvent.S3EXPORT)
        return context

    def test_func(self):
        return user_check(self.request.user)


class VideoDetail(UserPassesTestMixin,DetailView):
    model = Video

    def get_context_data(self, **kwargs):
        context = super(VideoDetail, self).get_context_data(**kwargs)
        max_frame_index = Frame.objects.all().filter(video=self.object).aggregate(Max('frame_index'))['frame_index__max']
        context['exports'] = TEvent.objects.all().filter(event_type=TEvent.EXPORT,video=self.object)
        context['s3_exports'] = TEvent.objects.all().filter(event_type=TEvent.S3EXPORT,video=self.object)
        context['annotation_count'] = Region.objects.all().filter(video=self.object,region_type=Region.ANNOTATION).count()
        if self.object.vdn_dataset:
            context['exportable_annotation_count'] = Region.objects.all().filter(video=self.object,vdn_dataset__isnull=True,region_type=Region.ANNOTATION).count()
        else:
            context['exportable_annotation_count'] = 0
        context['url'] = '{}{}/video/{}.mp4'.format(settings.MEDIA_URL,self.object.pk,self.object.pk)
        label_list = []
        show_all = self.request.GET.get('show_all_labels', False)
        context['label_list'] = label_list
        delta = 10000
        if context['object'].dataset:
            delta = 500
        if max_frame_index <= delta:
            context['frame_list'] = Frame.objects.all().filter(video=self.object).order_by('frame_index')
            context['detection_list'] = Region.objects.all().filter(video=self.object,region_type=Region.DETECTION)
            context['annotation_list'] = Region.objects.all().filter(video=self.object,region_type=Region.ANNOTATION)
            context['offset'] = 0
            context['limit'] = max_frame_index
        else:
            if self.request.GET.get('frame_index_offset', None) is None:
                offset = 0
            else:
                offset = int(self.request.GET.get('frame_index_offset'))
            limit = offset + delta
            context['offset'] = offset
            context['limit'] = limit
            context['frame_list'] = Frame.objects.all().filter(video=self.object,frame_index__gte=offset,frame_index__lte=limit).order_by('frame_index')
            context['detection_list'] = Region.objects.all().filter(video=self.object,parent_frame_index__gte=offset,parent_frame_index__lte=limit,region_type=Region.DETECTION)
            context['annotation_list'] = Region.objects.all().filter(video=self.object,parent_frame_index__gte=offset,parent_frame_index__lte=limit,region_type=Region.ANNOTATION)
            context['frame_index_offsets'] = [(k*delta,(k*delta)+delta) for k in range(int(math.ceil(max_frame_index / float(delta))))]
        context['frame_first'] = context['frame_list'].first()
        context['frame_last'] = context['frame_list'].last()
        if context['limit'] > max_frame_index:
            context['limit'] = max_frame_index
        context['max_frame_index'] = max_frame_index
        return context

    def test_func(self):
        return user_check(self.request.user)


class ClustersDetails(UserPassesTestMixin,DetailView):

    model = Clusters

    def get_context_data(self, **kwargs):
        context = super(ClustersDetails, self).get_context_data(**kwargs)
        context['coarse'] = []
        context['index_entries'] = [IndexEntries.objects.get(pk=k) for k in self.object.included_index_entries_pk]
        for k in ClusterCodes.objects.filter(clusters_id=self.object.pk).values('coarse_text').annotate(count=Count('coarse_text')):
            context['coarse'].append({'coarse_text':k['coarse_text'].replace(' ','_'),
                                      'count':k['count'],
                                      'first':ClusterCodes.objects.all().filter(clusters_id=self.object.pk,coarse_text=k['coarse_text']).first(),
                                      'last':ClusterCodes.objects.all().filter(clusters_id=self.object.pk,coarse_text=k['coarse_text']).last()
                                      })


        return context

    def test_func(self):
        return user_check(self.request.user)


class FrameList(UserPassesTestMixin,ListView):
    model = Frame

    def test_func(self):
        return user_check(self.request.user)


class FrameDetail(UserPassesTestMixin,DetailView):
    model = Frame

    def get_context_data(self, **kwargs):
        context = super(FrameDetail, self).get_context_data(**kwargs)
        context['detection_list'] = Region.objects.all().filter(frame=self.object,region_type=Region.DETECTION)
        context['annotation_list'] = Region.objects.all().filter(frame=self.object,region_type=Region.ANNOTATION)
        context['video'] = self.object.video
        context['url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,self.object.video.pk,self.object.frame_index)
        context['previous_frame'] = Frame.objects.filter(video=self.object.video,frame_index__lt=self.object.frame_index).order_by('-frame_index')[0:1]
        context['next_frame'] = Frame.objects.filter(video=self.object.video,frame_index__gt=self.object.frame_index).order_by('frame_index')[0:1]
        return context

    def test_func(self):
        return user_check(self.request.user)


class QueryList(UserPassesTestMixin,ListView):
    model = Query

    def test_func(self):
        return user_check(self.request.user)


class QueryDetail(UserPassesTestMixin,DetailView):
    model = Query

    def get_context_data(self, **kwargs):
        context = super(QueryDetail, self).get_context_data(**kwargs)
        context['inception'] = []
        context['facenet'] = []
        for r in QueryResults.objects.all().filter(query=self.object):
            if r.algorithm == 'facenet':
                context['facenet'].append((r.rank,r))
            else:
                context['inception'].append((r.rank,r))
        context['facenet'].sort()
        context['inception'].sort()
        if context['inception']:
            context['inception'] = zip(*context['inception'])[1]
        if context['facenet']:
            context['facenet'] = zip(*context['facenet'])[1]
        context['url'] = '{}queries/{}.png'.format(settings.MEDIA_URL,self.object.pk,self.object.pk)
        return context

    def test_func(self):
        return user_check(self.request.user)


class VDNDatasetDetail(UserPassesTestMixin,DetailView):
    model = VDNDataset

    def get_context_data(self, **kwargs):
        context = super(VDNDatasetDetail, self).get_context_data(**kwargs)
        context['video'] = Video.objects.get(vdn_dataset=context['object'])

    def test_func(self):
        return user_check(self.request.user)


@user_passes_test(user_check)
def search(request):
    if request.method == 'POST':
        count = request.POST.get('count')
        excluded_index_entries_pk = json.loads(request.POST.get('excluded_index_entries'))
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
        approximate = True if request.POST.get('approximate') == 'true' else False
        image_data_url = request.POST.get('image_url')
        query, dv = create_query(count,approximate,selected_indexers,excluded_index_entries_pk,image_data_url,request.user if request.user.is_authenticated else None)
        task_results = {}
        for visual_index_name,visual_index in settings.VISUAL_INDEXES.iteritems():
            task_name = visual_index['retriever_task']
            if visual_index_name in selected_indexers:
                task_results[visual_index_name] = app.send_task(task_name, args=[query.pk,],queue=settings.TASK_NAMES_TO_QUEUE[task_name])
        results = []
        results_detections = []
        time_out = False
        for visual_index_name,result in task_results.iteritems():
            entries = {}
            try:
                logging.info("Waiting for {}".format(visual_index_name))
                entries = result.get(timeout=120)
                print entries
                if entries:
                    if type(entries) is dict:
                        entries = entries
                    else:
                        entries = pickle.loads(base64.b64decode(entries))
            except TimeoutError:
                time_out = True
            except Exception, e:
                raise ValueError(e)
            if entries and settings.VISUAL_INDEXES[visual_index_name]['detection_specific']:
                for algo,rlist in entries.iteritems():
                    for r in rlist:
                        r['url'] = '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL,r['video_primary_key'],r['detection_primary_key'])
                        d = Region.objects.get(pk=r['detection_primary_key'])
                        r['result_detect'] = True
                        r['frame_primary_key'] = d.frame_id
                        r['result_type'] = 'detection'
                        r['detection'] = [{'pk': d.pk, 'name': d.object_name, 'confidence': d.confidence},]
                        results_detections.append(r)
            elif entries:
                for algo, rlist in entries.iteritems():
                    for r in rlist:
                        r['url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,r['video_primary_key'], r['frame_index'])
                        r['detections'] = [{'pk': d.pk, 'name': d.object_name, 'confidence': d.confidence} for d in
                                           Region.objects.filter(frame_id=r['frame_primary_key'])]
                        r['result_type'] = 'frame'
                        results.append(r)
        return JsonResponse(data={'task_id':"",'time_out':time_out,'primary_key':query.pk,'results':results,'results_detections':results_detections})


def home(request):
    return render(request, 'home.html', {})


@user_passes_test(user_check)
def index(request,query_pk=None,frame_pk=None,detection_pk=None):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated else None
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'],form.cleaned_data['name'],user=user,
                                 perform_scene_detection=form.cleaned_data['scene'],
                                 rate=form.cleaned_data['nth'],
                                 rescale=form.cleaned_data['rescale'] if 'rescale' in form.cleaned_data else 0)
            return redirect('video_list')
        else:
            raise ValueError
    else:
        form = UploadFileForm()
    context = { 'form' : form }
    context['indexes'] = settings.VISUAL_INDEXES
    if query_pk:
        previous_query = Query.objects.get(pk=query_pk)
        context['initial_url'] = '{}queries/{}.png'.format(settings.MEDIA_URL,query_pk)
    elif frame_pk:
        frame = Frame.objects.get(pk=frame_pk)
        context['initial_url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,frame.video.pk,frame.frame_index)
    elif detection_pk:
        detection = Region.objects.get(pk=detection_pk)
        context['initial_url'] = '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL,detection.video.pk, detection.pk)
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = Query.objects.count()
    context['index_entries_count'] = IndexEntries.objects.count()
    context['external_datasets_count'] = VDNDataset.objects.count()
    context['external_servers_count'] = VDNServer.objects.count()
    context['task_events_count'] = TEvent.objects.count()
    context['pending_tasks'] = TEvent.objects.all().filter(started=False).count()
    context['running_tasks'] = TEvent.objects.all().filter(started=True,completed=False).count()
    context['successful_tasks'] = TEvent.objects.all().filter(started=True,completed=True).count()
    context['errored_tasks'] = TEvent.objects.all().filter(errored=True).count()
    context['video_count'] = Video.objects.count() - context['query_count']
    context['index_entries'] = IndexEntries.objects.all()
    context['detection_count'] = Region.objects.all().filter(region_type=Region.DETECTION).count()
    context['annotation_count'] = Region.objects.all().filter(region_type=Region.ANNOTATION).count()
    return render(request, 'dashboard.html', context)

@user_passes_test(user_check)
def assign_video_labels(request):
    if request.method == 'POST':
        video = Video.objects.get(pk=request.POST.get('video_pk'))
        for k in request.POST.get('labels').split(','):
            if k.strip():
                dl = AppliedLabel()
                dl.video = video
                dl.label_name = k.strip()
                dl.source = dl.UI
                dl.save()
        return redirect('video_detail',pk=video.pk)
    else:
        raise NotImplementedError


@user_passes_test(user_check)
def annotate(request,frame_pk):
    context = {'frame':None, 'detection':None ,'existing':[]}
    frame = None
    frame = Frame.objects.get(pk=frame_pk)
    context['frame'] = frame
    context['initial_url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,frame.video.pk,frame.frame_index)
    context['previous_frame'] = Frame.objects.filter(video=frame.video,frame_index__lt=frame.frame_index).order_by('-frame_index')[0:1]
    context['next_frame'] = Frame.objects.filter(video=frame.video,frame_index__gt=frame.frame_index).order_by('frame_index')[0:1]
    context['detections'] = Region.objects.filter(frame=frame,region_type=Region.DETECTION)
    for d in Region.objects.filter(frame=frame):
        temp = {
            'x':d.x,
            'y':d.y,
            'h':d.h,
            'w':d.w,
            'pk':d.pk,
            'box_type':"detection" if d.region_type == d.DETECTION else 'annotation',
            'label': d.object_name,
            'full_frame': d.full_frame,
            'detection_pk':None
        }
        context['existing'].append(temp)
    context['existing'] = json.dumps(context['existing'])
    if request.method == 'POST':
        form = AnnotationForm(request.POST)
        if form.is_valid():
            applied_tags = form.cleaned_data['tags'].split(',') if form.cleaned_data['tags'] else []
            create_annotation(form, form.cleaned_data['object_name'], applied_tags, frame)
            return JsonResponse({'status': True})
        else:
            raise ValueError,form.errors
    return render(request, 'annotate.html', context)


@user_passes_test(user_check)
def annotate_entire_frame(request,frame_pk):
    frame = Frame.objects.get(pk=frame_pk)
    annotation = None
    if request.method == 'POST':
        if request.POST.get('metadata_text').strip() \
                or request.POST.get('metadata_json').strip() \
                or request.POST.get('object_name',None):
            annotation = Region()
            annotation.region_type = Region.ANNOTATION
            annotation.x = 0
            annotation.y = 0
            annotation.h = 0
            annotation.w = 0
            annotation.full_frame = True
            annotation.metadata_text = request.POST.get('metadata_text')
            annotation.metadata_json = request.POST.get('metadata_json')
            annotation.object_name = request.POST.get('object_name','frame_metadata')
            annotation.frame = frame
            annotation.video = frame.video
            annotation.save()
        for label_name in request.POST.get('tags').split(','):
            if label_name.strip():
                dl = AppliedLabel()
                dl.video = frame.video
                dl.frame = frame
                dl.label_name = label_name.strip()
                if annotation:
                    dl.region = annotation
                dl.source = dl.UI
                dl.save()
    return redirect("frame_detail",pk=frame.pk)


@user_passes_test(user_check)
def yt(request):
    if request.method == 'POST':
        form = YTVideoForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated else None
        if form.is_valid():
            handle_youtube_video(form.cleaned_data['name'],form.cleaned_data['url'],user=user,
                                 perform_scene_detection=form.cleaned_data['scene'],
                                 rate=form.cleaned_data['nth'],
                                 rescale=form.cleaned_data['rescale'] if 'rescale' in form.cleaned_data else 0)
        else:
            raise ValueError
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def export_video(request):
    if request.method == 'POST':
        pk = request.POST.get('video_id')
        video = Video.objects.get(pk=pk)
        export_method = request.POST.get('export_method')
        if video:
            if export_method == 's3':
                key = request.POST.get('key')
                region = request.POST.get('region')
                bucket = request.POST.get('bucket')
                s3export = TEvent()
                s3export.event_type = TEvent.S3EXPORT
                s3export.video = video
                s3export.key = key
                s3export.region = region
                s3export.bucket = bucket
                s3export.save()
                task_name = 'backup_video_to_s3'
                app.send_task(task_name, args=[s3export.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
            else:
                task_name = 'export_video_by_id'
                export_video_task = TEvent()
                export_video_task.event_type = TEvent.EXPORT
                export_video_task.video = video
                export_video_task.operation = task_name
                export_video_task.save()
                app.send_task(task_name, args=[export_video_task.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
        return redirect('video_list')
    else:
        raise NotImplementedError


@user_passes_test(user_check)
def coarse_code_detail(request,pk,coarse_code):
    coarse_code = coarse_code.replace('_',' ')
    context = {
               'code':coarse_code,
               'objects': ClusterCodes.objects.all().filter(coarse_text=coarse_code,clusters_id=pk)
               }
    return render(request,'coarse_code_details.html',context)


@user_passes_test(user_check)
def push(request,video_id):
    video = Video.objects.get(pk=video_id)
    if request.method == 'POST':
        push_type = request.POST.get('push_type')
        server = VDNServer.objects.get(pk=request.POST.get('server_pk'))
        token = request.POST.get('token_{}'.format(server.pk))
        server.last_token = token
        server.save()
        server_url = server.url
        if not server_url.endswith('/'):
            server_url += '/'
        headers = {'Authorization': 'Token {}'.format(server.last_token)}
        if push_type == 'annotation':
            new_vdn_dataset = create_child_vdn_dataset(video, server, headers)
            for key in request.POST:
                if key.startswith('annotation_') and request.POST[key]:
                    annotation = Region.objects.get(pk=int(key.split('annotation_')[1]))
                    data = {
                        'label':annotation.label,
                        'metadata_text':annotation.metadata_text,
                        'x':annotation.x,
                        'y':annotation.y,
                        'w':annotation.w,
                        'h':annotation.h,
                        'full_frame':annotation.full_frame,
                        'parent_frame_index':annotation.parent_frame_index,
                        'dataset_id':int(new_vdn_dataset.url.split('/')[-2]),
                    }
                    r = requests.post("{}/api/annotations/".format(server_url),data=data,headers=headers)
                    if r.status_code == 201:
                        annotation.vdn_dataset = new_vdn_dataset
                        annotation.save()
                    else:
                        raise ValueError
        elif push_type == 'dataset':
            key = request.POST.get('key')
            region = request.POST.get('region')
            bucket = request.POST.get('bucket')
            name = request.POST.get('name')
            description = request.POST.get('description')
            s3export = TEvent()
            s3export.event_type = TEvent.S3EXPORT
            s3export.video = video
            s3export.key = key
            s3export.region = region
            s3export.bucket = bucket
            s3export.save()
            create_root_vdn_dataset(s3export,server,headers,name,description)
            task_name = 'push_video_to_vdn_s3'
            app.send_task(task_name, args=[s3export.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
        else:
            raise NotImplementedError

    servers = VDNServer.objects.all()
    context = {'video':video, 'servers':servers}
    if video.vdn_dataset:
        context['annotations'] = Region.objects.all().filter(video=video, vdn_dataset__isnull=True,region_type=Region.ANNOTATION)
    else:
        context['annotations'] = Region.objects.all().filter(video=video,region_type=Region.ANNOTATION)
    return render(request,'push.html',context)


@user_passes_test(user_check)
def render_tasks(request,context):
    context['events'] = TEvent.objects.all()
    context['settings_queues'] = set(settings.TASK_NAMES_TO_QUEUE.values())
    task_list = []
    for k,v in settings.TASK_NAMES_TO_TYPE.iteritems():
        task_list.append({'name':k,
                          'type':v,
                          'queue':settings.TASK_NAMES_TO_QUEUE[k],
                          'edges':settings.POST_OPERATION_TASKS[k] if k in settings.POST_OPERATION_TASKS else []
                          })
    context['task_list'] = task_list
    context["videos"] = Video.objects.all().filter(parent_query__count__isnull=True)
    context['manual_tasks'] = settings.MANUAL_VIDEO_TASKS
    return render(request, 'tasks.html', context)


@user_passes_test(user_check)
def status(request):
    context = { }
    return render_status(request,context)


@user_passes_test(user_check)
def tasks(request):
    context = { }
    return render_tasks(request,context)


@user_passes_test(user_check)
def indexes(request):
    context = {
        'visual_index_list':settings.VISUAL_INDEXES.items(),
        'index_entries':IndexEntries.objects.all(),
        "videos" : Video.objects.all().filter(parent_query__count__isnull=True),
        "region_types" : Region.REGION_TYPES
    }
    if request.method == 'POST':
        if request.POST.get('visual_index_name') == 'inception':
            index_event = TEvent()
            index_event.operation = 'inception_index_regions_by_id'
            arguments ={
                'region_type__in': request.POST.getlist('region_type__in', []),
                'w__gte': int(request.POST.get('w__gte')),
                'h__gte': int(request.POST.get('h__gte'))
            }
            for optional_key in ['metadata_text__contains','object_name__contains','object_name']:
                if request.POST.get(optional_key,None):
                    arguments[optional_key] = request.POST.get(optional_key)
            for optional_key in ['h__lte','w__lte']:
                if request.POST.get(optional_key,None):
                    arguments[optional_key] = int(request.POST.get(optional_key))
            index_event.arguments_json = json.dumps(arguments)
            index_event.video_id = request.POST.get('video_id')
            index_event.save()
            app.send_task(name=index_event.operation, args=[index_event.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[index_event.operation])
        else:
            raise NotImplementedError
    return render(request, 'indexes.html', context)


@user_passes_test(user_check)
def detections(request):
    context = {}
    return render(request, 'detections.html', context)


@user_passes_test(user_check)
def textsearch(request):
    context = {}
    return render(request, 'textsearch.html', context)


@user_passes_test(user_check)
def clustering(request):
    context = {}
    context['clusters'] = Clusters.objects.all()
    context['algorithms'] = {k.algorithm for k in IndexEntries.objects.all()}
    context['index_entries'] = IndexEntries.objects.all()
    if request.method == 'POST':
        algorithm = request.POST.get('algorithm')
        v = request.POST.get('v')
        m = request.POST.get('m')
        components = request.POST.get('components')
        sub = request.POST.get('sub')
        excluded = request.POST.get('excluded_index_entries')
        c = Clusters()
        c.indexer_algorithm = algorithm
        c.included_index_entries_pk = [k.pk for k in IndexEntries.objects.all() if k.algorithm == c.indexer_algorithm]
        c.components = components
        c.sub = sub
        c.m = m
        c.v = v
        c.save()
        task_name = "perform_clustering"
        new_task = TEvent()
        new_task.clustering = c
        new_task.operation = task_name
        new_task.save()
        app.send_task(name=task_name, args=[new_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    return render(request, 'clustering.html', context)


@user_passes_test(user_check)
def delete_object(request):
    if request.method == 'POST':
        pk = request.POST.get('pk')
        if request.POST.get('object_type') == 'annotation':
            annotation = Region.objects.get(pk=pk)
            if annotation.region_type == Region.ANNOTATION:
                annotation.delete()
    return JsonResponse({'status':True})


@user_passes_test(user_check)
def import_dataset(request):
    if request.method == 'POST':
        url = request.POST.get('dataset_url')
        server = VDNServer.objects.get(pk=request.POST.get('server_pk'))
        user = request.user if request.user.is_authenticated else None
        import_vdn_dataset_url(server, url, user)
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def import_s3(request):
    if request.method == 'POST':
        keys = request.POST.get('key')
        region = request.POST.get('region')
        bucket = request.POST.get('bucket')
        for key in keys.strip().split('\n'):
            if key.strip():
                s3import = TEvent()
                s3import.event_type = TEvent.S3IMPORT
                s3import.key = key.strip()
                s3import.region = region
                s3import.bucket = bucket
                video = Video()
                user = request.user if request.user.is_authenticated else None
                if user:
                    video.uploader = user
                video.name = "pending S3 import {} s3://{}/{}".format(region,bucket,key)
                video.save()
                s3import.video = video
                s3import.save()
                task_name = 'import_video_from_s3'
                app.send_task(name=task_name, args=[s3import.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def video_send_task(request):
    if request.method == 'POST':
        video_id = int(request.POST.get('video_id'))
        task_name = request.POST.get('task_name')
        manual_event = TEvent()
        manual_event.video_id = video_id
        manual_event.save()
        app.send_task(name=task_name, args=[manual_event.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def external(request):
    if request.method == 'POST':
        pk = request.POST.get('server_pk')
        pull_vdn_dataset_list(pk)
    context = {
        'servers':VDNServer.objects.all(),
        'available':{ server:json.loads(server.last_response_datasets) for server in VDNServer.objects.all()},
        'vdn_datasets': VDNDataset.objects.all(),
    }
    return render(request, 'external_data.html', context)


@user_passes_test(user_check)
def retry_task(request,pk):
    event = TEvent.objects.get(pk=int(pk))
    context = {}
    if settings.TASK_NAMES_TO_TYPE[event.operation] == settings.VIDEO_TASK:
        new_event = TEvent()
        new_event.video_id = event.video_id
        new_event.arguments_json = event.arguments_json
        new_event.save()
        result = app.send_task(name=event.operation, args=[new_event.pk],queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        context['alert'] = "Operation {} on {} submitted".format(event.operation,event.video.name,queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        return render_tasks(request, context)
    elif settings.TASK_NAMES_TO_TYPE[event.operation] == settings.QUERY_TASK:
        return redirect("/requery/{}/".format(event.video.parent_query_id))
    else:
        raise NotImplementedError


@user_passes_test(user_check)
def render_status(request,context):
    context['video_count'] = Video.objects.count()
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = Query.objects.count()
    context['events'] = TEvent.objects.all()
    context['detection_count'] = Region.objects.all().filter(region_type=Region.DETECTION).count()
    context['settings_task_names_to_type'] = settings.TASK_NAMES_TO_TYPE
    context['settings_task_names_to_queue'] = settings.TASK_NAMES_TO_QUEUE
    context['settings_post_operation_tasks'] = settings.POST_OPERATION_TASKS
    context['settings_tasks'] = set(settings.TASK_NAMES_TO_QUEUE.keys())
    context['settings_queues'] = set(settings.TASK_NAMES_TO_QUEUE.values())
    try:
        context['indexer_log'] = file("logs/{}.log".format(settings.Q_INDEXER)).read()
    except:
        context['indexer_log'] = ""
    try:
        context['detector_log'] = file("logs/{}.log".format(settings.Q_DETECTOR)).read()
    except:
        context['detector_log'] = ""
    try:
        context['extract_log'] = file("logs/{}.log".format(settings.Q_EXTRACTOR)).read()
    except:
        context['extract_log'] = ""
    try:
        context['retriever_log'] = file("logs/{}.log".format(settings.Q_RETRIEVER)).read()
    except:
        context['retriever_log'] = ""
    try:
        context['face_retriever_log'] = file("logs/{}.log".format(settings.Q_FACE_RETRIEVER)).read()
    except:
        context['face_retriever_log'] = ""
    try:
        context['face_detector_log'] = file("logs/{}.log".format(settings.Q_FACE_DETECTOR)).read()
    except:
        context['face_detector_log'] = ""
    try:
        context['fab_log'] = file("logs/fab.log").read()
    except:
        context['fab_log'] = ""
    return render(request, 'status.html', context)


