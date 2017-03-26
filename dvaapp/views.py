from django.shortcuts import render,redirect
from django.conf import settings
from django.http import HttpResponse,JsonResponse,HttpResponseRedirect
import requests
import os,base64, json
from django.contrib.auth.decorators import login_required
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from .forms import UploadFileForm,YTVideoForm,AnnotationForm
from .models import Video,Frame,Detection,Query,QueryResults,TEvent,FrameLabel,IndexEntries,ExternalDataset, Annotation, VLabel
from .tasks import extract_frames,facenet_query_by_image,inception_query_by_image
from dva.celery import app
import serializers
from rest_framework import viewsets
from django.contrib.auth.models import User


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = serializers.UserSerializer


class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Video.objects.all()
    serializer_class = serializers.VideoSerializer


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
        for visual_index_name,result in task_results.iteritems():
            entries = result.get()
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
        return JsonResponse(data={'task_id':"",'primary_key':primary_key,'results':results,'results_detections':results_detections})


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
    context['external_datasets_count'] = ExternalDataset.objects.count()
    context['video_count'] = Video.objects.count() - context['query_count']
    context['detection_count'] = Detection.objects.count()
    context['annotation_count'] = Annotation.objects.count()
    return render(request, 'dashboard.html', context)


def annotate(request,query_pk=None,frame_pk=None,detection_pk=None):
    context = {'frame':None, 'detection':None ,'existing':[]}
    label_dict = {tag.label_name:tag.pk for tag in VLabel.objects.all()}
    context['available_tags'] = label_dict.keys()
    if query_pk:
        previous_query = Query.objects.get(pk=query_pk)
        context['initial_url'] = '/media/queries/{}.png'.format(query_pk)
    elif frame_pk:
        frame = Frame.objects.get(pk=frame_pk)
        context['frame'] = frame
        context['initial_url'] = '/media/{}/frames/{}.jpg'.format(frame.video.pk,frame.frame_index)
        context['previous_frame'] = Frame.objects.filter(video=frame.video,frame_index__lt=frame.frame_index).order_by('-frame_index')[0:1]
        context['next_frame'] = Frame.objects.filter(video=frame.video,frame_index__gt=frame.frame_index).order_by('frame_index')[0:1]
        for d in Detection.objects.filter(frame=frame):
            temp = {
                'x':d.x,
                'y':d.y,
                'h':d.h,
                'w':d.w,
                'pk':d.pk,
                'box_type':"detection",
                'name':d.object_name,
                'full_frame': False
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
                'name':d.name,
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
            annotation = Annotation()
            annotation.x = form.cleaned_data['x']
            annotation.y = form.cleaned_data['y']
            annotation.h = form.cleaned_data['h']
            annotation.w = form.cleaned_data['w']
            if form.cleaned_data['high_level']:
                annotation.full_frame = True
                annotation.x = 0
                annotation.y = 0
                annotation.h = 0
                annotation.w = 0
            annotation.name = form.cleaned_data['name']
            annotation.metadata_text = form.cleaned_data['metadata']
            if frame_pk:
                annotation.frame = frame
                annotation.video = frame.video
            annotation.save()
            if form.cleaned_data['tags']:
                applied_tags = json.loads(form.cleaned_data['tags'])
                if applied_tags:
                    annotation.label_count = len(applied_tags)
                    annotation.save()
                    for label_name in applied_tags:
                        applied_tag = FrameLabel()
                        applied_tag.label = ''
                        applied_tag.label_parent_id = label_dict[label_name]
                        applied_tag.annotation = annotation
                        applied_tag.frame_id = annotation.frame_id
                        applied_tag.video_id = annotation.video_id
                        applied_tag.label = label_name
                        applied_tag.save()
            return JsonResponse({'status': True})
        else:
            raise ValueError,form.errors
    return render(request, 'annotate.html', context)


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


def create_video_folders(video):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
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
    create_video_folders(video)
    primary_key = video.pk
    filename = f.name
    if filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
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


class VideoDetail(DetailView):
    model = Video

    def get_context_data(self, **kwargs):
        context = super(VideoDetail, self).get_context_data(**kwargs)
        context['frame_list'] = Frame.objects.all().filter(video=self.object)
        context['detection_list'] = Detection.objects.all().filter(video=self.object)
        context['annotation_list'] = Annotation.objects.all().filter(video=self.object)
        context['label_list'] = FrameLabel.objects.all().filter(video=self.object)
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
    context = {}
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


def external(request):
    context = {}
    return render(request, 'external_data.html', context)


def retry_task(request,pk):
    event = TEvent.objects.get(pk=int(pk))
    context = {}
    if event.operation != 'query_by_id':
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


