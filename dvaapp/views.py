from django.shortcuts import render,redirect
from django.conf import settings
from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
import requests
import os,base64, json
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from .forms import UploadFileForm,YTVideoForm,AnnotationForm
from .models import Video,Frame,Detection,Query,QueryResults,TEvent,IndexEntries,VDNDataset, Annotation, VLabel, Export, VDNServer, S3Export, S3Import, ClusterCodes, Clusters
from .tasks import extract_frames
from dva.celery import app
import serializers
from rest_framework import viewsets,mixins
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django.db.models import Count
from celery.exceptions import TimeoutError
import math
from django.db.models import Max,Avg,Sum

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = serializers.UserSerializer


class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Video.objects.all()
    serializer_class = serializers.VideoSerializer


class FrameViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Frame.objects.all()
    serializer_class = serializers.FrameSerializer
    filter_fields = ('frame_index', 'subdir', 'name', 'video')


class DetectionViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Detection.objects.all()
    serializer_class = serializers.DetectionSerializer
    filter_fields = ('video', 'frame', 'object_name')


class QueryViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Query.objects.all()
    serializer_class = serializers.QuerySerializer


class QueryResultsViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = QueryResults.objects.all()
    serializer_class = serializers.QueryResultsSerializer
    filter_fields = ('frame', 'video')


class AnnotationViewSet(mixins.ListModelMixin,mixins.RetrieveModelMixin,mixins.CreateModelMixin,mixins.DestroyModelMixin,viewsets.GenericViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = Annotation.objects.all()
    serializer_class = serializers.AnnotationSerializer
    filter_fields = ('video','frame')


class TEventViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = TEvent.objects.all()
    serializer_class = serializers.TEventSerializer
    filter_fields = ('video','operation')


class IndexEntriesViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = IndexEntries.objects.all()
    serializer_class = serializers.IndexEntriesSerializer
    filter_fields = ('video','algorithm','detection_name')


class VLabelViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VLabel.objects.all()
    serializer_class = serializers.VLabelSerializer


class VDNServerViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VDNServer.objects.all()
    serializer_class = serializers.VDNServerSerializer


class VDNDatasetViewSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticatedOrReadOnly,)
    queryset = VDNDataset.objects.all()
    serializer_class = serializers.VDNDatasetSerializer


def search(request):
    if request.method == 'POST':
        query = Query()
        count = request.POST.get('count')
        query.count = count
        excluded_index_entries_pk = json.loads(request.POST.get('excluded_index_entries'))
        if excluded_index_entries_pk:
            query.excluded_index_entries_pk = [int(k) for k in excluded_index_entries_pk]
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
        query.selected_indexers = selected_indexers
        query.save()
        primary_key = query.pk
        dv = Video()
        dv.name = 'query_{}'.format(query.pk)
        dv.dataset = True
        dv.query = True
        dv.parent_query = query
        dv.save()
        create_video_folders(dv)
        image_url = request.POST.get('image_url')
        image_data = base64.decodestring(image_url[22:])
        query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT,primary_key)
        query_frame_path = "{}/{}/frames/0.png".format(settings.MEDIA_ROOT,dv.pk)
        with open(query_path,'w') as fh:
            fh.write(image_data)
        with open(query_frame_path,'w') as fh:
            fh.write(image_data)
        task_results = {}
        user = request.user if request.user.is_authenticated() else None
        for visual_index_name,visual_index in settings.VISUAL_INDEXES.iteritems():
            task_name = visual_index['retriever_task']
            if visual_index_name in selected_indexers:
                task_results[visual_index_name] = app.send_task(task_name, args=[primary_key,],queue=settings.TASK_NAMES_TO_QUEUE[task_name])
        query.user = user
        query.save()
        results = []
        results_detections = []
        time_out = False
        for visual_index_name,result in task_results.iteritems():
            try:
                entries = result.get(timeout=120)
            except TimeoutError:
                time_out = True
                entries = {}
            if entries and settings.VISUAL_INDEXES[visual_index_name]['detection_specific']:
                for algo,rlist in entries.iteritems():
                    for r in rlist:
                        r['url'] = '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL,r['video_primary_key'],r['detection_primary_key'])
                        d = Detection.objects.get(pk=r['detection_primary_key'])
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
                                           Detection.objects.filter(frame_id=r['frame_primary_key'])]
                        r['result_type'] = 'frame'
                        results.append(r)
        return JsonResponse(data={'task_id':"",'time_out':time_out,'primary_key':primary_key,'results':results,'results_detections':results_detections})


def index(request,query_pk=None,frame_pk=None,detection_pk=None):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated() else None
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'],form.cleaned_data['name'],user=user)
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
        detection = Detection.objects.get(pk=detection_pk)
        context['initial_url'] = '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL,detection.video.pk, detection.pk)
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = Query.objects.count()
    context['index_entries_count'] = IndexEntries.objects.count()
    context['external_datasets_count'] = VDNDataset.objects.count()
    context['external_servers_count'] = VDNServer.objects.count()
    context['task_events_count'] = TEvent.objects.count()
    context['video_count'] = Video.objects.count() - context['query_count']
    context['index_entries'] = IndexEntries.objects.all()
    context['detection_count'] = Detection.objects.count()
    context['annotation_count'] = Annotation.objects.count()
    return render(request, 'dashboard.html', context)


def annotate(request,frame_pk):
    context = {'frame':None, 'detection':None ,'existing':[]}
    frame = None
    frame = Frame.objects.get(pk=frame_pk)
    label_dict = {tag.label_name: tag.pk for tag in VLabel.objects.filter(video=frame.video).all()}
    context['available_tags'] = label_dict.keys()
    context['frame'] = frame
    context['initial_url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,frame.video.pk,frame.frame_index)
    context['previous_frame'] = Frame.objects.filter(video=frame.video,frame_index__lt=frame.frame_index).order_by('-frame_index')[0:1]
    context['next_frame'] = Frame.objects.filter(video=frame.video,frame_index__gt=frame.frame_index).order_by('frame_index')[0:1]
    context['detections'] = Detection.objects.filter(frame=frame)
    for d in Detection.objects.filter(frame=frame):
        temp = {
            'x':d.x,
            'y':d.y,
            'h':d.h,
            'w':d.w,
            'pk':d.pk,
            'box_type':"detection",
            'label':d.object_name,
            'full_frame': False,
            'detection_pk':None
        }
        context['existing'].append(temp)
    for d in Annotation.objects.filter(frame=frame):
        temp = {
            'x':d.x,
            'y':d.y,
            'h':d.h,
            'w':d.w,
            'pk': d.pk,
            'box_type':"annotation",
            'label':d.label,
            'full_frame':d.full_frame
        }
        context['existing'].append(temp)
    context['existing'] = json.dumps(context['existing'])
    if request.method == 'POST':
        form = AnnotationForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['tags']:
                applied_tags = json.loads(form.cleaned_data['tags'])
                applied_tags = applied_tags if applied_tags else ['metadata',]
                for label_name in applied_tags:
                    create_annotation(form, label_name, label_dict, frame)
            return JsonResponse({'status': True})
        else:
            raise ValueError,form.errors
    return render(request, 'annotate.html', context)


def annotate_entire_frame(request,frame_pk):
    frame = Frame.objects.get(pk=frame_pk)
    label_dict = {tag.label_name: tag.pk for tag in VLabel.objects.filter(video=frame.video).all()}
    if request.method == 'POST':
        applied_tags = request.POST.getlist('tags')
        applied_tags = applied_tags if applied_tags else ['metadata', ]
        for label_name in applied_tags:
            annotation = Annotation()
            annotation.x = 0
            annotation.y = 0
            annotation.h = 0
            annotation.w = 0
            annotation.full_frame = True
            annotation.metadata_text = request.POST.get('metadata_text')
            annotation.metadata_json = request.POST.get('metadata_json')
            annotation.label = label_name
            if label_name in label_dict:
                annotation.label_parent_id = label_dict[label_name]
            annotation.frame = frame
            annotation.video = frame.video
            annotation.save()
    return redirect("frame_detail",pk=frame.pk)


def create_annotation(form,label_name,label_dict,frame):
    annotation = Annotation()
    if form.cleaned_data['high_level']:
        annotation.full_frame = True
        annotation.x = 0
        annotation.y = 0
        annotation.h = 0
        annotation.w = 0
    else:
        annotation.full_frame = False
        annotation.x = form.cleaned_data['x']
        annotation.y = form.cleaned_data['y']
        annotation.h = form.cleaned_data['h']
        annotation.w = form.cleaned_data['w']
    annotation.metadata_text = form.cleaned_data['metadata_text']
    annotation.metadata_json = form.cleaned_data['metadata_json']
    annotation.label = label_name
    if label_name in label_dict:
        annotation.label_parent_id = label_dict[label_name]
    annotation.frame = frame
    annotation.video = frame.video
    annotation.save()


def yt(request):
    if request.method == 'POST':
        form = YTVideoForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated() else None
        if form.is_valid():
            handle_youtube_video(form.cleaned_data['name'],form.cleaned_data['url'],user=user)
        else:
            raise ValueError
    else:
        raise NotImplementedError
    return redirect('video_list')


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
                s3export = S3Export()
                s3export.video = video
                s3export.key = key
                s3export.region = region
                s3export.bucket = bucket
                s3export.save()
                task_name = 'backup_video_to_s3'
                app.send_task(task_name, args=[s3export.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
            else:
                task_name = 'export_video_by_id'
                app.send_task(task_name, args=[pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
        return redirect('video_list')
    else:
        raise NotImplementedError


def create_video_folders(video,create_subdirs=True):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
    if create_subdirs:
        os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/detections/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, video.pk))


def handle_youtube_video(name,url,extract=True,user=None):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.url = url
    video.youtube_video = True
    video.save()
    create_video_folders(video)
    if extract:
        extract_frames.apply_async(args=[video.pk], queue=settings.Q_EXTRACTOR)
    return video


def handle_uploaded_file(f,name,extract=True,user=None):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.save()
    primary_key = video.pk
    filename = f.name
    filename = filename.lower()
    if filename.endswith('.dva_export.zip'):
        create_video_folders(video, create_subdirs=False)
        with open('{}/{}/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        app.send_task(name=task_name, args=[primary_key,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video,create_subdirs=True)
        with open('{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            extract_frames.apply_async(args=[primary_key],queue=settings.Q_EXTRACTOR)
    else:
        raise ValueError,"Extension {} not allowed".format(filename.split('.')[-1])
    return video


class VideoList(ListView):
    model = Video
    paginate_by = 100

    def get_context_data(self, **kwargs):
        context = super(VideoList, self).get_context_data(**kwargs)
        context['exports'] = Export.objects.all()
        context['s3_exports'] = S3Export.objects.all()
        return context


class VideoDetail(DetailView):
    model = Video

    def get_context_data(self, **kwargs):
        context = super(VideoDetail, self).get_context_data(**kwargs)
        max_frame_index = Frame.objects.all().filter(video=self.object).aggregate(Max('frame_index'))['frame_index__max']
        context['exports'] = Export.objects.all().filter(video=self.object)
        context['s3_exports'] = S3Export.objects.all().filter(video=self.object)
        context['annotation_count'] = Annotation.objects.all().filter(video=self.object).count()
        if self.object.vdn_dataset:
            context['exportable_annotation_count'] = Annotation.objects.all().filter(video=self.object,vdn_dataset__isnull=True).count()
        else:
            context['exportable_annotation_count'] = 0
        context['label_count'] = VLabel.objects.all().filter(video=self.object).count()
        context['url'] = '{}/{}/video/{}.mp4'.format(settings.MEDIA_URL,self.object.pk,self.object.pk)
        label_list = []
        show_all = self.request.GET.get('show_all_labels', False)
        if context['label_count'] < 100 or show_all:
            for k in VLabel.objects.all().filter(video=context['object']):
                label_list.append({'label_name': k.label_name,'created': k.created,'pk': k.pk,'source': k.get_source_display(),'count':Annotation.objects.filter(label_parent=k).count()})
        else:
            context['label_count_warning'] = True
        context['label_list'] = label_list
        delta = 10000
        if context['object'].dataset:
            delta = 1000
        if max_frame_index <= delta:
            context['frame_list'] = Frame.objects.all().filter(video=self.object).order_by('frame_index')
            context['detection_list'] = Detection.objects.all().filter(video=self.object)
            context['annotation_list'] = Annotation.objects.all().filter(video=self.object)
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
            context['detection_list'] = Detection.objects.all().filter(video=self.object,parent_frame_index__gte=offset,parent_frame_index__lte=limit)
            context['annotation_list'] = Annotation.objects.all().filter(video=self.object,parent_frame_index__gte=offset,parent_frame_index__lte=limit)
            context['frame_index_offsets'] = [(k*delta,(k*delta)+delta) for k in range(int(math.ceil(max_frame_index / float(delta))))]
        context['frame_first'] = context['frame_list'].first()
        context['frame_last'] = context['frame_list'].last()
        if context['limit'] > max_frame_index:
            context['limit'] = max_frame_index
        context['max_frame_index'] = max_frame_index
        return context


class ClustersDetails(DetailView):

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


def coarse_code_detail(request,pk,coarse_code):
    coarse_code = coarse_code.replace('_',' ')
    context = {
               'code':coarse_code,
               'objects': ClusterCodes.objects.all().filter(coarse_text=coarse_code,clusters_id=pk)
               }
    return render(request,'coarse_code_details.html',context)

class QueryList(ListView):
    model = Query


class QueryDetail(DetailView):
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
        context['url'] = '{}/queries/{}.png'.format(settings.MEDIA_URL,self.object.pk,self.object.pk)
        return context


class FrameList(ListView):
    model = Frame


def create_child_vdn_dataset(parent_video,server,headers):
    server_url = server.url
    if not server_url.endswith('/'):
        server_url += '/'
    new_dataset = {'root': False,
                   'parent_url': parent_video.vdn_dataset.url ,
                   'description':'automatically created child'}
    r = requests.post("{}api/datasets/".format(server_url), data=new_dataset, headers=headers)
    if r.status_code == 201:
        vdn_dataset = VDNDataset()
        vdn_dataset.url = r.json()['url']
        vdn_dataset.root = False
        vdn_dataset.response = r.text
        vdn_dataset.server = server
        vdn_dataset.parent_local = parent_video.vdn_dataset
        vdn_dataset.save()
        return vdn_dataset
    else:
        raise ValueError,"{} {} {} {}".format("{}api/datasets/".format(server_url),headers,r.status_code,new_dataset)


def create_root_vdn_dataset(s3export,server,headers,name,description):
    new_dataset = {'root': True,
                   'aws_requester_pays':True,
                   'aws_region':s3export.region,
                   'aws_bucket':s3export.bucket,
                   'aws_key':s3export.key,
                   'name': name,
                   'description': description
                   }
    server_url = server.url
    if not server_url.endswith('/'):
        server_url += '/'
    r = requests.post("{}api/datasets/".format(server_url), data=new_dataset, headers=headers)
    if r.status_code == 201:
        vdn_dataset = VDNDataset()
        vdn_dataset.url = r.json()['url']
        vdn_dataset.root = True
        vdn_dataset.response = r.text
        vdn_dataset.server = server
        vdn_dataset.save()
        s3export.video.vdn_dataset = vdn_dataset
        return vdn_dataset
    else:
        raise ValueError,"Could not crated dataset"


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
                    annotation = Annotation.objects.get(pk=int(key.split('annotation_')[1]))
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
            s3export = S3Export()
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
        context['annotations'] = Annotation.objects.all().filter(video=video, vdn_dataset__isnull=True)
    else:
        context['annotations'] = Annotation.objects.all().filter(video=video)
    return render(request,'push.html',context)


class FrameDetail(DetailView):
    model = Frame

    def get_context_data(self, **kwargs):
        context = super(FrameDetail, self).get_context_data(**kwargs)
        context['detection_list'] = Detection.objects.all().filter(frame=self.object)
        context['annotation_list'] = Annotation.objects.all().filter(frame=self.object)
        context['video'] = self.object.video
        context['url'] = '{}/{}/frames/{}.jpg'.format(settings.MEDIA_URL,self.object.video.pk,self.object.frame_index)
        context['previous_frame'] = Frame.objects.filter(video=self.object.video,frame_index__lt=self.object.frame_index).order_by('-frame_index')[0:1]
        context['next_frame'] = Frame.objects.filter(video=self.object.video,frame_index__gt=self.object.frame_index).order_by('frame_index')[0:1]
        label_dict = {tag.label_name: tag.pk for tag in VLabel.objects.filter(video=self.object.video).all()}
        context['available_tags'] = label_dict.keys()
        return context


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
    """
      // create an array with nodes
      var nodes = [
        {id: 1, label: 'Node 1'},
        {id: 2, label: 'Node 2'},
        {id: 3, label: 'Node 3:\nLeft-Aligned', font: {'face': 'Monospace', align: 'left'}},
        {id: 4, label: 'Node 4'},
        {id: 5, label: 'Node 5\nLeft-Aligned box', shape: 'box',
         font: {'face': 'Monospace', align: 'left'}}
      ];

      // create an array with edges
      var edges = [
        {from: 1, to: 2, label: 'middle',     font: {align: 'middle'}},
        {from: 1, to: 3, label: 'top',        font: {align: 'top'}},
        {from: 2, to: 4, label: 'horizontal', font: {align: 'horizontal'}},
        {from: 2, to: 5, label: 'bottom',     font: {align: 'bottom'}}
      ];
    """
    taskid = {}
    nodes = []
    edges = []
    for i,k in enumerate(settings.TASK_NAMES_TO_TYPE.iteritems()):
        nodes.append({'id':i,'label':"{} type {}".format(k[0],k[1])})
        taskid[k[0]] = i
    for initial,subs in settings.POST_OPERATION_TASKS.iteritems():
        for v in subs:
            edges.append({"from":taskid[initial],'to':taskid[v],'label':settings.TASK_NAMES_TO_QUEUE[v],'arrows':'to'})
    context["nodes"] = json.dumps(nodes)
    context["edges"] = json.dumps(edges)
    context["videos"] = Video.objects.all().filter(parent_query__count__isnull=True)
    context['manual_tasks'] = settings.MANUAL_VIDEO_TASKS
    return render(request, 'tasks.html', context)


def status(request):
    context = { }
    return render_status(request,context)


def tasks(request):
    context = { }
    return render_tasks(request,context)



def indexes(request):
    context = {
        'visual_index_list':settings.VISUAL_INDEXES.items(),
        'index_entries':IndexEntries.objects.all()
    }

    return render(request, 'indexes.html', context)


def annotations(request):
    query = Annotation.objects.all().values('label_parent_id').annotate(
        total=Count('pk'),
        frame_count=Count('frame',distinct=True),
        video_count=Count('video',distinct=True)).order_by('total')
    query_result = []
    for k in query:
        label = VLabel.objects.get(pk=k['label_parent_id'])
        query_result.append({'label_name':label.label_name,
                             'created': label.created,
                             'hidden': label.hidden,
                             'pk': label.pk,
                             'vdn_dataset': label.vdn_dataset,
                             'source':label.get_source_display(),
                             'total':k['total'],
                             'frame_count': k['frame_count'],
                             'video_count':k['video_count']})
    context = {'vlabels':VLabel.objects.all(),'label_stats':query_result,
               'annotations_count':Annotation.objects.all().count(),'labels_count':VLabel.objects.all().count(),}
    return render(request, 'annotations.html', context)


def detections(request):
    context = {}
    return render(request, 'detections.html', context)


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
        app.send_task(name=task_name, args=[c.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    return render(request, 'clustering.html', context)


def delete_object(request):
    if request.method == 'POST':
        pk = request.POST.get('pk')
        if request.POST.get('object_type') == 'annotation':
            annotation = Annotation.objects.get(pk=pk)
            annotation.delete()
    return JsonResponse({'status':True})


def create_dataset(d,server):
    dataset = VDNDataset()
    dataset.server = server
    dataset.name = d['name']
    dataset.description = d['description']
    dataset.download_url = d['download_url']
    dataset.url = d['url']
    dataset.aws_bucket = d['aws_bucket']
    dataset.aws_key = d['aws_key']
    dataset.aws_region = d['aws_region']
    dataset.aws_requester_pays = d['aws_requester_pays']
    dataset.organization_url = d['organization']['url']
    dataset.response = json.dumps(d)
    dataset.save()
    return dataset


def import_dataset(request):
    if request.method == 'POST':
        url = request.POST.get('dataset_url')
        r = requests.get(url)
        response = r.json()
        server = VDNServer.objects.get(pk=request.POST.get('server_pk'))
        vdn_dataset = create_dataset(response,server)
        vdn_dataset.save()
        video = Video()
        user = request.user if request.user.is_authenticated() else None
        if user:
            video.uploader = user
        video.name = vdn_dataset.name
        video.vdn_dataset = vdn_dataset
        video.save()
        primary_key = video.pk
        create_video_folders(video, create_subdirs=False)
        task_name = 'import_video_by_id'
        app.send_task(name=task_name, args=[primary_key, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise NotImplementedError
    return redirect('video_list')


def import_s3(request):
    if request.method == 'POST':
        s3import = S3Import()
        key = request.POST.get('key')
        region = request.POST.get('region')
        bucket = request.POST.get('bucket')
        s3import.key = key
        s3import.region = region
        s3import.bucket = bucket
        video = Video()
        user = request.user if request.user.is_authenticated() else None
        if user:
            video.uploader = user
        video.name = "pending S3 import {} s3://{}/{}".format(region,key,bucket)
        video.save()
        s3import.video = video
        s3import.save()
        create_video_folders(video, create_subdirs=False)
        task_name = 'import_video_from_s3'
        app.send_task(name=task_name, args=[s3import.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise NotImplementedError
    return redirect('video_list')


def video_send_task(request):
    if request.method == 'POST':
        video_id = int(request.POST.get('video_id'))
        task_name = request.POST.get('task_name')
        app.send_task(name=task_name, args=[video_id, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise NotImplementedError
    return redirect('video_list')


def create_label(request):
    if request.method == 'POST':
        video_id = request.POST.get('video_pk')
        name = request.POST.get('name')
        multiple = request.POST.get('video_pk')
        video = Video.objects.get(pk=video_id)
        if multiple:
            for k in name.strip().split(','):
                label, created = VLabel.objects.get_or_create(label_name=k, source=VLabel.UI, video=video)
        else:
            label,created = VLabel.objects.get_or_create(label_name=name,source=VLabel.UI,video=video)
    else:
        raise NotImplementedError
    return redirect('video_detail',pk=video_id)


def delete_label(request):
    if request.method == 'POST':
        for key in request.POST:
            if key.startswith('delete_label_pk_') and request.POST[key]:
                label = VLabel.objects.get(pk=int(key.split('_')[-1]))
                Annotation.objects.filter(label_parent=label).delete()
                label.delete()
        video_id = request.POST.get('video_pk')
        video = Video.objects.get(pk=video_id)
    else:
        raise NotImplementedError
    return redirect('video_detail',pk=video_id)


def external(request):
    if request.method == 'POST':
        pk = request.POST.get('server_pk')
        server = VDNServer.objects.get(pk=pk)
        r = requests.get("{}api/datasets/".format(server.url))
        response = r.json()
        datasets = []
        for d in response['results']:
            datasets.append(d)

        while response['next']:
            r = request.get("{}api/datasets/".format(server))
            response = r.json()
            for d in response['results']:
                datasets.append(d)
        server.last_response_datasets = json.dumps(datasets)
        server.save()
    context = {
        'servers':VDNServer.objects.all(),
        'available':{ server:json.loads(server.last_response_datasets) for server in VDNServer.objects.all()},
        'vdn_datasets': VDNDataset.objects.all(),
    }
    return render(request, 'external_data.html', context)


class VDNDatasetDetail(DetailView):
    model = VDNDataset

    def get_context_data(self, **kwargs):
        context = super(VDNDatasetDetail, self).get_context_data(**kwargs)
        context['video'] = Video.objects.get(vdn_dataset=context['object'])


def retry_task(request,pk):
    event = TEvent.objects.get(pk=int(pk))
    context = {}
    if settings.TASK_NAMES_TO_TYPE[event.operation] == settings.VIDEO_TASK:
        result = app.send_task(name=event.operation, args=[event.video_id],queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        context['alert'] = "Operation {} on {} submitted".format(event.operation,event.video.name,queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        return render_tasks(request, context)
    elif settings.TASK_NAMES_TO_TYPE[event.operation] == settings.QUERY_TASK:
        return redirect("/requery/{}/".format(event.video.parent_query_id))
    else:
        raise NotImplementedError


def render_status(request,context):
    context['video_count'] = Video.objects.count()
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = Query.objects.count()
    context['events'] = TEvent.objects.all()
    context['detection_count'] = Detection.objects.count()
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


