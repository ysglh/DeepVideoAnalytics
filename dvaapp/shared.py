import os, json, requests, base64, logging
from models import Video, TEvent, AppliedLabel, Region, VDNDataset, VDNServer, VDNDetector, CustomDetector, DeletedVideo
from django.conf import settings
from dva.celery import app
from django_celery_results.models import TaskResult
from celery.result import AsyncResult
from collections import defaultdict
from operations import processing
from models import DVAPQL
import boto3
from celery.exceptions import TimeoutError


def refresh_task_status():
    for t in TEvent.objects.all().filter(started=True, completed=False, errored=False):
        try:
            tr = TaskResult.objects.get(task_id=t.task_id)
        except TaskResult.DoesNotExist:
            pass
        else:
            if tr.status == 'FAILURE':
                t.errored = True
                t.save()


def create_video_folders(video, create_subdirs=True):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
    if create_subdirs:
        os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/segments/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/regions/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/transforms/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, video.pk))


def create_detector_folders(detector, create_subdirs=True):
    try:
        os.mkdir('{}/detectors/{}'.format(settings.MEDIA_ROOT, detector.pk))
    except:
        pass


def create_indexer_folders(indexer, create_subdirs=True):
    try:
        os.mkdir('{}/indexers/{}'.format(settings.MEDIA_ROOT, indexer.pk))
    except:
        pass


def create_annotator_folders(annotator, create_subdirs=True):
    try:
        os.mkdir('{}/annotators/{}'.format(settings.MEDIA_ROOT, annotator.pk))
    except:
        pass


def delete_video_object(video_pk,deleter,garbage_collection=True):
    video = Video.objects.get(pk=video_pk)
    deleted = DeletedVideo()
    deleted.name = video.name
    deleted.deleter = deleter
    deleted.uploader = video.uploader
    deleted.url = video.url
    deleted.description = video.description
    deleted.original_pk = video_pk
    deleted.save()
    video.delete()
    if garbage_collection:
        delete_task = TEvent()
        delete_task.arguments = {'video_pk': video_pk}
        delete_task.operation = 'delete_video_by_id'
        delete_task.save()
        queue = settings.TASK_NAMES_TO_QUEUE[delete_task.operation]
        _ = app.send_task(name=delete_task.operation, args=[delete_task.pk], queue=queue)


def handle_uploaded_file(f, name, extract=True, user=None, rate=30, rescale=0):
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
        with open('{}/{}/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]),
                  'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        video.save()
        operation = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=operation, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[operation])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        with open('{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]),
                  'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            p = processing.DVAPQLProcess()
            if video.dataset:
                query = {
                    'process_type':DVAPQL.PROCESS,
                    'tasks':[
                        {
                            'arguments':{'rate': rate, 'rescale': rescale,'next_tasks':settings.DEFAULT_PROCESSING_PLAN},
                            'video_id':video.pk,
                            'operation': 'extract_frames',
                        }
                    ]
                }
            else:
                query = {
                    'process_type':DVAPQL.PROCESS,
                    'tasks':[
                        {
                            'arguments':{'next_tasks':[
                                             {'operation':'decode_video',
                                               'arguments':{
                                                   'rate': rate,
                                                   'rescale': rescale,
                                                   'next_tasks':settings.DEFAULT_PROCESSING_PLAN
                                               }
                                              }
                                            ]},
                            'video_id':video.pk,
                            'operation': 'segment_video',
                        }
                    ]
                }
            p.create_from_json(j=query,user=user)
            p.launch()
    else:
        raise ValueError, "Extension {} not allowed".format(filename.split('.')[-1])
    return video


def handle_downloaded_file(downloaded, video, name, extract=True, user=None, rate=30, rescale=0):
    video.name = name
    video.save()
    filename = downloaded.split('/')[-1]
    if filename.endswith('.dva_export.zip'):
        create_video_folders(video, create_subdirs=False)
        os.rename(downloaded, '{}/{}/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        video.save()
        operation = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=operation, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[operation])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        os.rename(downloaded,
                  '{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            p = processing.DVAPQLProcess()
            if video.dataset:
                query = {
                    'process_type':DVAPQL.PROCESS,
                    'tasks':[
                        {
                            'arguments':{'rate': rate, 'rescale': rescale,'next_tasks':settings.DEFAULT_PROCESSING_PLAN},
                            'video_id':video.pk,
                            'operation': 'extract_frames',
                        }
                    ]
                }
            else:
                query = {
                    'process_type':DVAPQL.PROCESS,
                    'tasks':[
                        {
                            'arguments':{'next_tasks':[
                                             {'operation':'decode_video',
                                              'arguments':{
                                                   'rate': rate,
                                                   'rescale': rescale,
                                                   'next_tasks':settings.DEFAULT_PROCESSING_PLAN
                                               }
                                              }
                                            ]},
                            'video_id':video.pk,
                            'operation': 'segment_video',
                        }
                    ]
                }
            p.create_from_json(j=query,user=user)
            p.launch()
    else:
        raise ValueError, "Extension {} not allowed".format(filename.split('.')[-1])
    return video


def create_annotation(form, object_name, labels, frame):
    annotation = Region()
    annotation.object_name = object_name
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
    annotation.frame = frame
    annotation.video = frame.video
    annotation.region_type = Region.ANNOTATION
    annotation.save()
    for l in labels:
        if l.strip():
            dl = AppliedLabel()
            dl.video = annotation.video
            dl.frame = annotation.frame
            dl.region = annotation
            dl.label_name = l.strip()
            dl.source = dl.UI
            dl.save()


def handle_youtube_video(name, url, extract=True, user=None, rate=30, rescale=0):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.url = url
    video.youtube_video = True
    video.save()
    if extract:
        p = processing.DVAPQLProcess()
        query = {
            'process_type': DVAPQL.PROCESS,
            'tasks': [
                {
                    'arguments': {'next_tasks': [
                        {'operation': 'decode_video',
                         'arguments': {
                             'rate': rate,
                             'rescale': rescale,
                             'next_tasks': settings.DEFAULT_PROCESSING_PLAN
                         }
                         }
                    ]},
                    'video_id': video.pk,
                    'operation': 'segment_video',
                }
            ]
        }
        p.create_from_json(j=query, user=user)
        p.launch()
    return video


def create_child_vdn_dataset(parent_video, server, headers):
    server_url = server.url
    if not server_url.endswith('/'):
        server_url += '/'
    new_dataset = {'root': False,
                   'parent_url': parent_video.vdn_dataset.url,
                   'description': 'automatically created child'}
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
        raise ValueError, "{} {} {} {}".format("{}api/datasets/".format(server_url), headers, r.status_code,
                                               new_dataset)


def create_root_vdn_dataset(region, bucket, key, server, headers, name, description):
    new_dataset = {'root': True,
                   'aws_requester_pays': True,
                   'aws_region': region,
                   'aws_bucket': bucket,
                   'aws_key': key,
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
        return vdn_dataset
    else:
        raise ValueError, "Could not crated dataset"


def pull_vdn_list(pk):
    """
    Pull list of datasets from configured VDN servers
    """
    server = VDNServer.objects.get(pk=pk)
    datasets = []
    detectors = []
    r = requests.get("{}vdn/api/datasets/".format(server.url))
    response = r.json()
    for d in response['results']:
        datasets.append(d)
    while response['next']:
        r = requests.get(response['next'])
        response = r.json()
        for d in response['results']:
            datasets.append(d)
    r = requests.get("{}vdn/api/detectors/".format(server.url))
    response = r.json()
    for d in response['results']:
        detectors.append(d)
    while response['next']:
        r = requests.get(response['next'])
        response = r.json()
        for d in response['results']:
            detectors.append(d)
    server.last_response_datasets = json.dumps(datasets)
    server.last_response_detectors = json.dumps(detectors)
    server.save()
    return server, datasets, detectors




def create_dataset(d, server):
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


def import_vdn_dataset_url(server, url, user):
    r = requests.get(url)
    response = r.json()
    vdn_dataset = create_dataset(response, server)
    vdn_dataset.save()
    video = Video()
    if user:
        video.uploader = user
    video.name = vdn_dataset.name
    video.vdn_dataset = vdn_dataset
    video.save()
    if vdn_dataset.download_url:
        operation = 'import_vdn_file'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.operation = operation
        import_video_task.save()
        app.send_task(name=operation, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[operation])
    elif vdn_dataset.aws_key and vdn_dataset.aws_bucket:
        operation = 'import_vdn_s3'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.operation = operation
        import_video_task.save()
        app.send_task(name=operation, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[operation])
    else:
        raise NotImplementedError


def create_vdn_detector(d, server):
    vdn_detector = VDNDetector()
    vdn_detector.server = server
    vdn_detector.name = d['name']
    vdn_detector.description = d['description']
    vdn_detector.download_url = d['download_url']
    vdn_detector.url = d['url']
    vdn_detector.aws_bucket = d['aws_bucket']
    vdn_detector.aws_key = d['aws_key']
    vdn_detector.aws_region = d['aws_region']
    vdn_detector.aws_requester_pays = d['aws_requester_pays']
    vdn_detector.organization_url = d['organization']['url']
    vdn_detector.response = json.dumps(d)
    vdn_detector.save()
    return vdn_detector


def import_vdn_detector_url(server, url, user):
    r = requests.get(url)
    response = r.json()
    vdn_detector = create_vdn_detector(response, server)
    detector = CustomDetector()
    detector.name = vdn_detector.name
    detector.vdn_detector = vdn_detector
    detector.save()
    if vdn_detector.download_url:
        operation = 'import_vdn_detector_file'
        import_vdn_detector_task = TEvent()
        import_vdn_detector_task.operation = operation
        import_vdn_detector_task.arguments = {'detector_pk': detector.pk}
        import_vdn_detector_task.save()
        app.send_task(name=operation, args=[import_vdn_detector_task.pk, ],
                      queue=settings.TASK_NAMES_TO_QUEUE[operation])
    elif vdn_detector.aws_key and vdn_detector.aws_bucket:
        raise NotImplementedError
    else:
        raise NotImplementedError


def create_detector_dataset(object_names, labels):
    class_distribution = defaultdict(int)
    rboxes = defaultdict(list)
    rboxes_set = defaultdict(set)
    frames = {}
    class_names = {k: i for i, k in enumerate(labels.union(object_names))}
    i_class_names = {i: k for k, i in class_names.items()}
    for r in Region.objects.all().filter(object_name__in=object_names):
        frames[r.frame_id] = r.frame
        if r.pk not in rboxes_set[r.frame_id]:
            rboxes[r.frame_id].append((class_names[r.object_name], r.x, r.y, r.x + r.w, r.y + r.h))
            rboxes_set[r.frame_id].add(r.pk)
            class_distribution[r.object_name] += 1
    for l in AppliedLabel.objects.all().filter(label_name__in=labels):
        frames[l.frame_id] = l.frame
        if l.region:
            r = l.region
            if r.pk not in rboxes_set[r.frame_id]:
                rboxes[l.frame_id].append((class_names[l.label_name], r.x, r.y, r.x + r.w, r.y + r.h))
                rboxes_set[r.frame_id].add(r.pk)
                class_distribution[l.label_name] += 1
    return class_distribution, class_names, rboxes, rboxes_set, frames, i_class_names

