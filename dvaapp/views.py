from django.shortcuts import render,redirect
from django.conf import settings
from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
import requests
import os,base64, json
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from .forms import UploadFileForm,YTVideoForm,AnnotationForm,VLabelForm
from .models import Video,Frame,Detection,Query,QueryResults,TEvent,IndexEntries,VDNDataset, Annotation, VLabel, Export, VDNServer
from .tasks import extract_frames
from dva.celery import app
import serializers
from rest_framework import viewsets,mixins
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django.db.models import Count
from celery.exceptions import TimeoutError


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


def search(request):
    if request.method == 'POST':
        query = Query()
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
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
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
                        r['url'] = '/media/{}/detections/{}.jpg'.format(r['video_primary_key'],r['detection_primary_key'])
                        d = Detection.objects.get(pk=r['detection_primary_key'])
                        r['result_detect'] = True
                        r['frame_primary_key'] = d.frame_id
                        r['result_type'] = 'detection'
                        r['detection'] = [{'pk': d.pk, 'name': d.object_name, 'confidence': d.confidence},]
                        results_detections.append(r)
            elif entries:
                for algo, rlist in entries.iteritems():
                    for r in rlist:
                        r['url'] = '/media/{}/frames/{}.jpg'.format(r['video_primary_key'], r['frame_index'])
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
        else:
            raise ValueError
    else:
        form = UploadFileForm()
    context = { 'form' : form }
    context['indexes'] = settings.VISUAL_INDEXES
    if query_pk:
        previous_query = Query.objects.get(pk=query_pk)
        context['initial_url'] = '/media/queries/{}.png'.format(query_pk)
    elif frame_pk:
        frame = Frame.objects.get(pk=frame_pk)
        context['initial_url'] = '/media/{}/frames/{}.jpg'.format(frame.video.pk,frame.frame_index)
    elif detection_pk:
        detection = Detection.objects.get(pk=detection_pk)
        context['initial_url'] = '/media/{}/detections/{}.jpg'.format(detection.video.pk, detection.pk)
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = Query.objects.count()
    context['index_entries_count'] = IndexEntries.objects.count()
    context['external_datasets_count'] = VDNDataset.objects.count()
    context['video_count'] = Video.objects.count() - context['query_count']
    context['detection_count'] = Detection.objects.count()
    context['annotation_count'] = Annotation.objects.count()
    return render(request, 'dashboard.html', context)


def annotate(request,query_pk=None,frame_pk=None,detection_pk=None):
    context = {'frame':None, 'detection':None ,'existing':[]}
    label_dict = {tag.label_name:tag.pk for tag in VLabel.objects.filter(source=VLabel.UI).all()}
    context['available_tags'] = label_dict.keys()
    frame = None
    if query_pk:
        previous_query = Query.objects.get(pk=query_pk)
        context['initial_url'] = '/media/queries/{}.png'.format(query_pk)
    elif frame_pk:
        frame = Frame.objects.get(pk=frame_pk)
        context['frame'] = frame
        context['initial_url'] = '/media/{}/frames/{}.jpg'.format(frame.video.pk,frame.frame_index)
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
                'detection_pk': d.detection_id,
                'full_frame':d.full_frame
            }
            context['existing'].append(temp)
        context['existing'] = json.dumps(context['existing'])
    elif detection_pk:
        detection = Detection.objects.get(pk=detection_pk)
        context['detection'] = detection
        context['initial_url'] = '/media/{}/detections/{}.jpg'.format(detection.video.pk, detection.pk)
    if request.method == 'POST':
        form = AnnotationForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['tags']:
                applied_tags = json.loads(form.cleaned_data['tags'])
                if form.cleaned_data['metadata'].strip():
                    create_annotation(form, "metadata", label_dict, frame_pk, frame)
                if applied_tags:
                    for label_name in applied_tags:
                        create_annotation(form, label_name, label_dict, frame_pk, frame)
            return JsonResponse({'status': True})
        else:
            raise ValueError,form.errors
    return render(request, 'annotate.html', context)


def create_annotation(form,label_name,label_dict,frame_pk,frame):
    annotation = Annotation()
    if form.cleaned_data['high_level']:
        annotation.full_frame = True
        annotation.x = 0
        annotation.y = 0
        annotation.h = 0
        annotation.w = 0
    else:
        annotation.x = form.cleaned_data['x']
        annotation.y = form.cleaned_data['y']
        annotation.h = form.cleaned_data['h']
        annotation.w = form.cleaned_data['w']
    if form.cleaned_data['detection'] >= 0:
        detection = Detection.objects.get(pk=int(form.cleaned_data['detection']))
        annotation.detection=detection
    annotation.metadata_text = form.cleaned_data['metadata']
    annotation.label = label_name
    if label_name in label_dict:
        annotation.label_parent_id = label_dict[label_name]
    if frame_pk:
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
    return redirect('app')


def export_video(request):
    if request.method == 'POST':
        pk = request.POST.get('video_id')
        video = Video.objects.get(pk=pk)
        task_name = 'export_video_by_id'
        if video:
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


def handle_uploaded_file(f,name,extract=True,user=None):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.save()
    primary_key = video.pk
    filename = f.name
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


class VideoList(ListView):
    model = Video
    paginate_by = 100

    def get_context_data(self, **kwargs):
        context = super(VideoList, self).get_context_data(**kwargs)
        context['exports'] = Export.objects.all()
        return context


class VideoDetail(DetailView):
    model = Video

    def get_context_data(self, **kwargs):
        context = super(VideoDetail, self).get_context_data(**kwargs)
        context['frame_list'] = Frame.objects.all().filter(video=self.object)
        context['detection_list'] = Detection.objects.all().filter(video=self.object)
        context['annotation_list'] = Annotation.objects.all().filter(video=self.object)
        context['label_list'] = Annotation.objects.all().filter(video=self.object)
        context['exports'] = Export.objects.all().filter(video=self.object)
        context['url'] = '{}/{}/video/{}.mp4'.format(settings.MEDIA_URL,self.object.pk,self.object.pk)
        return context

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

        return context


def status(request):
    context = { }
    return render_status(request,context)


def indexes(request):
    context = {
        'visual_index_list':settings.VISUAL_INDEXES.items(),
        'index_entries':IndexEntries.objects.all()
    }

    return render(request, 'indexes.html', context)


def annotations(request):
    if request.method == 'POST':
        form = VLabelForm(request.POST)
        if form.is_valid():
            label = form.save(commit=False)
            label.save()
    form = VLabelForm()
    query = Annotation.objects.all().values('label_parent_id').annotate(
        total=Count('pk'),
        frame_count=Count('frame',distinct=True),
        video_count=Count('video',distinct=True)).order_by('total')
    query_result = []
    for k in query:
        label = VLabel.objects.get(pk=k['label_parent_id'])
        query_result.append({'label_name':label.label_name,
                             'source':label.get_source_display(),
                             'total':k['total'],
                             'frame_count': k['frame_count'],
                             'video_count':k['video_count']})
    context = {'form':form,'vlabels':VLabel.objects.all(),'label_stats':query_result,
               'annotations_count':Annotation.objects.all().count(),'labels_count':VLabel.objects.all().count(),}
    return render(request, 'annotations.html', context)


def detections(request):
    context = {}
    return render(request, 'detections.html', context)


def delete_object(request):
    if request.method == 'POST':
        pk = request.POST.get('pk')
        if request.POST.get('object_type') == 'annotation':
            annotation = Annotation.objects.get(pk=pk)
            annotation.delete()
    return JsonResponse({'status':True})


def create_dataset(d,server,existing):
    dataset = VDNDataset()
    dataset.server = server
    dataset.name = d['name']
    dataset.description = d['description']
    dataset.download_url = d['download_url']
    dataset.aws_bucket = d['aws_bucket']
    dataset.aws_key = d['aws_key']
    dataset.aws_region = d['aws_region']
    dataset.aws_requester_pays = d['aws_requester_pays']
    dataset.organization_url = d['organization']['url']
    if not ("{}/{}".format(dataset.organization_url, dataset.name) in existing):
        dataset.save()


def import_dataset(request):
    if request.method == 'POST':
        vdn_dataset_pk = request.POST.get('vdn_dataset_pk')
        vdn_dataset = VDNDataset.objects.get(pk=vdn_dataset_pk)
        video = Video()
        user = request.user if request.user.is_authenticated() else None
        if user:
            video.uploader = user
        video.name = vdn_dataset.name
        video.save()
        primary_key = video.pk
        vdn_dataset.child_video = video
        vdn_dataset.save()
        create_video_folders(video, create_subdirs=False)
        output_filename = "{}/{}/{}.zip".format(settings.MEDIA_ROOT,primary_key,primary_key)
        if 'www.dropbox.com' in vdn_dataset.download_url and not vdn_dataset.download_url.endswith('?dl=1'):
            r = requests.get(vdn_dataset.download_url+'?dl=1')
        else:
            r = requests.get(vdn_dataset.download_url)
        with open(output_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        r.close()
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        app.send_task(name=task_name, args=[primary_key, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    return redirect('video_list')


def external(request):
    if request.method == 'POST':
        pk = request.POST.get('server_pk')
        server = VDNServer.objects.get(pk=pk)
        r = requests.get("{}api/datasets/".format(server.url))
        existing = ["{}/{}".format(k.organization_url,k.name) for k in VDNDataset.objects.all().filter(server=server)]
        response = r.json()
        for d in response['results']:
            create_dataset(d,server,existing)
        while response['next']:
            r = request.get("{}api/datasets/".format(server))
            response = r.json()
            for d in response['results']:
                create_dataset(d,server,existing)
    context = {
        'servers':VDNServer.objects.all(),
        'datasets':VDNDataset.objects.all()
    }
    return render(request, 'external_data.html', context)


def retry_task(request,pk):
    event = TEvent.objects.get(pk=int(pk))
    context = {}
    if settings.TASK_NAMES_TO_TYPE[event.operation] == settings.VIDEO_TASK:
        result = app.send_task(name=event.operation, args=[event.video_id],queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        context['alert'] = "Operation {} on {} submitted".format(event.operation,event.video.name,queue=settings.TASK_NAMES_TO_QUEUE[event.operation])
        return render_status(request, context)
    else:
        return redirect("/requery/{}/".format(event.video.parent_query_id))


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


