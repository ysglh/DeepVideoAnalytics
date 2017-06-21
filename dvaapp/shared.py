import os, json, requests, base64, logging
from models import Video, TEvent, AppliedLabel, Region, Frame, VDNDataset, VDNServer, Query, VDNDetector, \
    CustomDetector, QueryResults, IndexerQuery
from django.conf import settings
from dva.celery import app
from celery.result import AsyncResult
from collections import defaultdict
import boto3
from celery.exceptions import TimeoutError


def refresh_task_status():
    for t in TEvent.objects.all().filter(started=True, completed=False, errored=False):
        if AsyncResult(t.id).status == 'FAILURE':
            t.errored = True
            t.save()


def create_video_folders(video, create_subdirs=True):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
    if create_subdirs:
        os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/segments/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/detections/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, video.pk))


def create_detector_folders(detector, create_subdirs=True):
    try:
        os.mkdir('{}/models/{}'.format(settings.MEDIA_ROOT, detector.pk))
    except:
        pass


def handle_uploaded_file(f, name, extract=True, user=None, perform_scene_detection=True, rate=30, rescale=0):
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
        task_name = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
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
            extract_frames_task = TEvent()
            extract_frames_task.arguments_json = json.dumps({'perform_scene_detection': perform_scene_detection,
                                                             'rate': rate,
                                                             'rescale': rescale})
            extract_frames_task.video = video
            task_name = 'extract_frames_by_id'
            extract_frames_task.operation = task_name
            extract_frames_task.save()
            app.send_task(name=task_name, args=[extract_frames_task.pk, ],
                          queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise ValueError, "Extension {} not allowed".format(filename.split('.')[-1])
    return video


def handle_downloaded_file(downloaded, video, name, extract=True, user=None, perform_scene_detection=True, rate=30,
                           rescale=0, ):
    video.name = name
    video.save()
    filename = downloaded.split('/')[-1]
    if filename.endswith('.dva_export.zip'):
        create_video_folders(video, create_subdirs=False)
        os.rename(downloaded, '{}/{}/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        os.rename(downloaded,
                  '{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            extract_frames_task = TEvent()
            extract_frames_task.arguments_json = json.dumps(
                {'perform_scene_detection': perform_scene_detection, 'rate': rate, 'rescale': rescale})
            extract_frames_task.video = video
            task_name = 'extract_frames_by_id'
            extract_frames_task.operation = task_name
            extract_frames_task.save()
            app.send_task(name=task_name, args=[extract_frames_task.pk, ],
                          queue=settings.TASK_NAMES_TO_QUEUE[task_name])
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


def handle_youtube_video(name, url, extract=True, user=None, perform_scene_detection=True, rate=30, rescale=0):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.url = url
    video.youtube_video = True
    video.save()
    task_name = 'extract_frames_by_id'
    extract_frames_task = TEvent()
    extract_frames_task.video = video
    extract_frames_task.operation = task_name
    extract_frames_task.arguments_json = json.dumps({'perform_scene_detection': perform_scene_detection,
                                                     'rate': rate,
                                                     'rescale': rescale})
    extract_frames_task.save()
    if extract:
        app.send_task(name=task_name, args=[extract_frames_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
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


def create_root_vdn_dataset(s3export, server, headers, name, description):
    new_dataset = {'root': True,
                   'aws_requester_pays': True,
                   'aws_region': s3export.region,
                   'aws_bucket': s3export.bucket,
                   'aws_key': s3export.key,
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


class QueryProcessing(object):

    def __init__(self):
        self.query = None
        self.indexer_queries = []
        self.task_results = {}
        self.context = {}
        self.visual_indexes = settings.VISUAL_INDEXES

    def store_and_create_video_object(self):
        dv = Video()
        dv.name = 'query_{}'.format(self.query.pk)
        dv.dataset = True
        dv.query = True
        dv.parent_query = self.query
        dv.save()
        if settings.HEROKU_DEPLOY:
            query_key = "queries/{}.png".format(self.query.pk)
            query_frame_key = "{}/frames/0.png".format(dv.pk)
            s3 = boto3.resource('s3')
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_key, Body=self.query.image_data)
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_frame_key, Body=self.query.image_data)
        else:
            query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.query.pk)
            query_frame_path = "{}/{}/frames/0.png".format(settings.MEDIA_ROOT, dv.pk)
            create_video_folders(dv)
            with open(query_path, 'w') as fh:
                fh.write(self.query.image_data)
            with open(query_frame_path, 'w') as fh:
                fh.write(self.query.image_data)

    def create_from_view(self, count, approximate, selected, excluded_pks, image_data_url, user=None):
        self.query = Query()
        self.query.approximate = approximate
        if not (user is None):
            self.query.user = user
        image_data = base64.decodestring(image_data_url[22:])
        self.query.image_data = image_data
        self.query.save()
        self.store_and_create_video_object()
        for k in selected:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k
            iq.count = count
            iq.excluded_index_entries_pk = [int(k) for k in excluded_pks]
            iq.approximate = approximate
            self.indexer_queries.append(k)
            self.indexer_queries[k].save()

        return self.query

    def load_from_db(self,query_pk):
        pass

    def send_tasks(self):
        for iq in self.indexer_queries:
            task_name = self.visual_indexes[iq.algorithm]['retriever_task']
            queue_name = settings.TASK_NAMES_TO_QUEUE[task_name]
            self.task_results[iq.algorithm] = app.send_task(task_name, args=[self.query.pk, ], queue=queue_name)
            self.context[iq.algorithm] = []

    def wait_and_collect(self,timeout=120):
        for visual_index_name, result in self.task_results.iteritems():
            try:
                logging.info("Waiting for {}".format(visual_index_name))
                _ = result.get(timeout=120)
            except TimeoutError:
                time_out = True
            except Exception, e:
                raise ValueError(e)
        for r in QueryResults.objects.all().filter(query=self.query):
            self.context[r.algorithm].append((r.rank,
                                         {'url': '{}{}/detections/{}.jpg'.format(settings.MEDIA_URL, r.video_id,
                                                                                 r.detection_id) if r.detection_id else '{}{}/frames/{}.jpg'.format(
                                             settings.MEDIA_URL, r.video_id, r.frame_id),
                                          'result_type': "Region" if r.detection_id else "Frame",
                                          'rank':r.rank,
                                          'frame_id': r.frame_id,
                                          'frame_index': r.frame.frame_index,
                                          'video_id': r.video_id,
                                          'video_name': r.video.name}))
        for k, v in context.iteritems():
            if v:
                context[k].sort()
                context[k] = zip(*v)[1]


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
        task_name = 'import_vdn_file'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.operation = task_name
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif vdn_dataset.aws_key and vdn_dataset.aws_bucket:
        task_name = 'import_vdn_s3'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.operation = task_name
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
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
        task_name = 'import_vdn_detector_file'
        import_vdn_detector_task = TEvent()
        import_vdn_detector_task.operation = task_name
        import_vdn_detector_task.arguments_json = json.dumps({'detector_pk': detector.pk})
        import_vdn_detector_task.save()
        app.send_task(name=task_name, args=[import_vdn_detector_task.pk, ],
                      queue=settings.TASK_NAMES_TO_QUEUE[task_name])
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


def perform_query(count, approximate, selected_indexers, excluded_index_entries_pk, image_data_url, user):
    qp = QueryProcessing()
    query = qp.create_from_view(count, approximate, selected_indexers, excluded_index_entries_pk, image_data_url, user)
    qp.send_tasks()

    return {'task_id': "", 'primary_key': query.pk, 'results': context, 'url':'{}queries/{}.png'.format(settings.MEDIA_URL, query.pk)}