import os, json, requests, shutil, zipfile, cStringIO, base64
from dvaapp.models import Video, TEvent,  VDNServer, Label, RegionLabel, DeepModel, Retriever, DVAPQL, Region, Frame, \
    QueryRegion, QueryRegionResults,QueryResults
from django.conf import settings
from django_celery_results.models import TaskResult
from collections import defaultdict
from dvaapp import processing
from dvaapp import serializers
from PIL import Image
import defaults


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


def delete_video_object(video_pk,deleter):
    p = processing.DVAPQLProcess()
    query = {
        'process_type': DVAPQL.PROCESS,
        'tasks': [
            {
                'arguments': {'video_pk': video_pk,'deleter_pk':deleter.pk},
                'operation': 'perform_deletion',
            }
        ]
    }
    p.create_from_json(j=query, user=deleter)
    p.launch()


def handle_uploaded_file(f, name, extract=True, user=None, rate=None, rescale=None):
    if rate is None:
        rate = defaults.DEFAULT_RATE
    if rescale is None:
        rescale = defaults.DEFAULT_RESCALE
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.save()
    filename = f.name
    filename = filename.lower()
    if filename.endswith('.dva_export.zip'):
        video.create_directory(create_subdirs=False)
        with open('{}/{}/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]),
                  'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        video.save()
        p = processing.DVAPQLProcess()
        query = {
            'process_type': DVAPQL.PROCESS,
            'tasks': [
                {
                    'arguments': {'source':'LOCAL'},
                    'video_id': video.pk,
                    'operation': 'perform_import',
                }
            ]
        }
        p.create_from_json(j=query, user=user)
        p.launch()
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        video.create_directory(create_subdirs=True)
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
                            'arguments':{'rescale': rescale,
                                         'next_tasks':defaults.DEFAULT_PROCESSING_PLAN_DATASET},
                            'video_id':video.pk,
                            'operation': 'perform_dataset_extraction',
                        }
                    ]
                }
            else:
                query = {
                    'process_type':DVAPQL.PROCESS,
                    'tasks':[
                        {
                            'arguments':{
                                'next_tasks':[
                                             {'operation':'perform_video_decode',
                                               'arguments':{
                                                   'segments_batch_size': defaults.DEFAULT_SEGMENTS_BATCH_SIZE,
                                                   'rate': rate,
                                                   'rescale': rescale,
                                                   'next_tasks':defaults.DEFAULT_PROCESSING_PLAN_VIDEO
                                               }
                                              }
                                            ]},
                            'video_id':video.pk,
                            'operation': 'perform_video_segmentation',
                        }
                    ]
                }
            p.create_from_json(j=query,user=user)
            p.launch()
    else:
        raise ValueError, "Extension {} not allowed".format(filename.split('.')[-1])
    return video


def handle_downloaded_file(downloaded, video, name):
    video.name = name
    video.save()
    filename = downloaded.split('/')[-1]
    if filename.endswith('.dva_export.zip'):
        video.create_directory(create_subdirs=False)
        os.rename(downloaded, '{}/{}/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        video.save()
        import_local(video)
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        video.create_directory(create_subdirs=True)
        os.rename(downloaded,
                  '{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
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
    annotation.text = form.cleaned_data['text']
    annotation.metadata = form.cleaned_data['metadata']
    annotation.frame = frame
    annotation.video = frame.video
    annotation.region_type = Region.ANNOTATION
    annotation.save()
    for lname in labels:
        if lname.strip():
            dl, _ = Label.objects.get_or_create(name=lname, set="UI")
            rl = RegionLabel()
            rl.video = annotation.video
            rl.frame = annotation.frame
            rl.region = annotation
            rl.label = dl
            rl.save()


def handle_video_url(name, url, user = None):
    return Video.objects.create(name=name,url=url,youtube_video=True,uploader=user)


def pull_vdn_list(pk):
    """
    Pull list of datasets and models from configured VDN servers.
    Currently just uses default response since the VDN is disabled.
    """
    server = VDNServer.objects.get(pk=pk)
    datasets = server.last_response_datasets
    detectors = server.last_response_detectors
    try:
        r = requests.get("{}vdn/api/vdn_datasets/".format(server.url))
        response = r.json()
        for d in response['results']:
            datasets.append(d)
        while response['next']:
            r = requests.get(response['next'])
            response = r.json()
            for d in response['results']:
                datasets.append(d)
        r = requests.get("{}vdn/api/vdn_detectors/".format(server.url))
        response = r.json()
        for d in response['results']:
            detectors.append(d)
        while response['next']:
            r = requests.get(response['next'])
            response = r.json()
            for d in response['results']:
                detectors.append(d)
        server.last_response_datasets = datasets
        server.last_response_detectors = detectors
        server.save()
    except:
        pass
    return server, datasets, detectors


def import_vdn_dataset_url(server, url, user, cached_response):
    response = None
    try:
        r = requests.get(url)
        response = r.json()
    except:
        pass
    if not response:
        response = cached_response
    video = Video()
    video.description = "import from {} : {} ".format(server.url,response['description'])
    if user:
        video.uploader = user
    video.name = response['name']
    video.save()
    if response['download_url']:
        p = processing.DVAPQLProcess()
        query = {
            'process_type': DVAPQL.PROCESS,
            'tasks': [
                {
                    'arguments': {'source':'VDN_URL','url':response['download_url']},
                    'video_id': video.pk,
                    'operation': 'perform_import',
                }
            ]
        }
        p.create_from_json(j=query, user=user)
        p.launch()
    elif response['aws_key'] and response['aws_bucket']:
        p = processing.DVAPQLProcess()
        query = {
            'process_type': DVAPQL.PROCESS,
            'tasks': [
                {
                    'arguments': {'source':'VDN_S3','key':response['aws_key'],'bucket':response['aws_bucket']},
                    'video_id': video.pk,
                    'operation': 'perform_import',
                }
            ]
        }
        p.create_from_json(j=query, user=user)
        p.launch()
    else:
        raise NotImplementedError


def import_vdn_detector_url(server, url, user, cached_response):
    response = None
    try:
        r = requests.get(url)
        response = r.json()
    except:
        pass
    if not response:
        response = cached_response
    detector = DeepModel()
    detector.model_type = DeepModel.DETECTOR
    detector.name = response['name']
    detector.detector_type = response.get('detector_type',DeepModel.YOLO)
    detector.save()
    if response.get('download_url',False):
        p = processing.DVAPQLProcess()
        query = {
            'process_type': DVAPQL.PROCESS,
            'tasks': [
                {
                    'arguments': {'detector_pk': detector.pk,'download_url':response['download_url']},
                    'operation': 'perform_detector_import',
                }
            ]
        }
        p.create_from_json(j=query, user=user)
        p.launch()
    elif response.get('aws_key',False) and response.get('aws_bucket',False):
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
    for dl in Label.objects.filter(name__in=labels):
        lname = dl.name
        for l in RegionLabel.all().objects.filter(label=dl):
            frames[l.frame_id] = l.frame
            if l.region:
                r = l.region
                if r.pk not in rboxes_set[r.frame_id]:
                    rboxes[l.frame_id].append((class_names[lname], r.x, r.y, r.x + r.w, r.y + r.h))
                    rboxes_set[r.frame_id].add(r.pk)
                    class_distribution[lname] += 1
    return class_distribution, class_names, rboxes, rboxes_set, frames, i_class_names


def import_local(dv):
    video_id = dv.pk
    video_obj = Video.objects.get(pk=video_id)
    zipf = zipfile.ZipFile("{}/{}/{}.zip".format(settings.MEDIA_ROOT, video_id, video_id), 'r')
    zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, video_id))
    zipf.close()
    video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, video_id)
    old_key = None
    for k in os.listdir(video_root_dir):
        unzipped_dir = "{}{}".format(video_root_dir, k)
        if os.path.isdir(unzipped_dir):
            for subdir in os.listdir(unzipped_dir):
                shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
            shutil.rmtree(unzipped_dir)
            break
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, video_id)) as input_json:
        video_json = json.load(input_json)
    importer = serializers.VideoImporter(video=video_obj, json=video_json, root_dir=video_root_dir)
    importer.import_video()
    source_zip = "{}/{}.zip".format(video_root_dir, video_obj.pk)
    os.remove(source_zip)


def create_query_from_request(p, request):
    """
    Create JSON object representing the query from request received from Dashboard.
    :param request:
    :return:
    """
    query_json = {'process_type': DVAPQL.QUERY}
    count = request.POST.get('count')
    generate_tags = request.POST.get('generate_tags')
    selected_indexers = json.loads(request.POST.get('selected_indexers',"[]"))
    selected_detectors = json.loads(request.POST.get('selected_detectors',"[]"))
    query_json['image_data_b64'] = request.POST.get('image_url')[22:]
    query_json['tasks'] = []
    indexer_tasks = defaultdict(list)
    if generate_tags and generate_tags != 'false':
        query_json['tasks'].append({'operation': 'perform_analysis',
                                    'arguments': {'analyzer': 'tagger','target': 'query',}
                                    })

    if selected_indexers:
        for k in selected_indexers:
            indexer_pk, retriever_pk = k.split('_')
            indexer_tasks[int(indexer_pk)].append(int(retriever_pk))
    for i in indexer_tasks:
        di = DeepModel.objects.get(pk=i,model_type=DeepModel.INDEXER)
        rtasks = []
        for r in indexer_tasks[i]:
            rtasks.append({'operation': 'perform_retrieval', 'arguments': {'count': int(count), 'retriever_pk': r}})
        query_json['tasks'].append(
            {
                'operation': 'perform_indexing',
                'arguments': {
                    'index': di.name,
                    'target': 'query',
                    'next_tasks': rtasks
                }

            }
        )
    if selected_detectors:
        for d in selected_detectors:
            dd = DeepModel.objects.get(pk=int(d),model_type=DeepModel.DETECTOR)
            if dd.name == 'textbox':
                query_json['tasks'].append({'operation': 'perform_detection',
                                            'arguments': {'detector_pk': int(d),
                                                          'target': 'query',
                                                          'next_tasks': [{
                                                              'operation': 'perform_analysis',
                                                              'arguments': {'target': 'query_regions',
                                                                            'analyzer': 'crnn',
                                                                            'filters': {'event_id': '__parent_event__'}
                                                                            }
                                                          }]
                                                          }
                                            })
            elif dd.name == 'face':
                dr = Retriever.objects.get(name='facenet',algorithm=Retriever.EXACT)
                query_json['tasks'].append({'operation': 'perform_detection',
                                            'arguments': {'detector_pk': int(d),
                                                          'target': 'query',
                                                          'next_tasks': [{
                                                              'operation': 'perform_indexing',
                                                              'arguments': {'target': 'query_regions',
                                                                            'index': 'facenet',
                                                                            'filters': {'event_id': '__parent_event__'},
                                                                            'next_tasks':[{
                                                                                'operation':'perform_retrieval',
                                                                                'arguments':{'retriever_pk':dr.pk,
                                                                                             'filters':{'event_id': '__parent_event__'},
                                                                                             'target':'query_region_index_vectors',
                                                                                             'count':10}
                                                                            }]}
                                                          }]
                                                          }
                                            })
            else:
                query_json['tasks'].append({'operation': 'perform_detection',
                                            'arguments': {'detector_pk': int(d), 'target': 'query', }})
    user = request.user if request.user.is_authenticated else None
    p.create_from_json(query_json, user)
    return p.process


def collect(p):
    context = {'results': defaultdict(list), 'regions': []}
    rids_to_names = {}
    for rd in QueryRegion.objects.all().filter(query=p.process):
        rd_json = get_query_region_json(rd)
        for r in QueryRegionResults.objects.filter(query=p.process, query_region=rd):
            gather_results(r, rids_to_names, rd_json['results'])
        context['regions'].append(rd_json)
    for r in QueryResults.objects.all().filter(query=p.process):
        gather_results(r, rids_to_names, context['results'])
    for k, v in context['results'].iteritems():
        if v:
            context['results'][k].sort()
            context['results'][k] = zip(*v)[1]
    for rd in context['regions']:
        for k, v in rd['results'].iteritems():
            if v:
                rd['results'][k].sort()
                rd['results'][k] = zip(*v)[1]
    return context


def gather_results(r,rids_to_names,results):
    name = get_retrieval_event_name(r,rids_to_names)
    results[name].append((r.rank, get_result_json(r)))


def get_url(r):
    if r.detection_id:
        dd = r.detection
        if dd.materialized:
            return '{}{}/regions/{}.jpg'.format(settings.MEDIA_URL, r.video_id,r.detection_id)
        else:
            frame_url = get_frame_url(r)
            if frame_url.startswith('http'):
                response = requests.get(frame_url)
                img = Image.open(cStringIO.StringIO(response.content))
            else:
                img = Image.open('{}/{}/frames/{}.jpg'.format(settings.MEDIA_ROOT,r.video_id,r.frame.frame_index))
            cropped = img.crop((dd.x, dd.y, dd.x + dd.w, dd.y + dd.h))
            buffer = cStringIO.StringIO()
            cropped.save(buffer, format="JPEG")
            return "data:image/jpeg;base64, {}".format(base64.b64encode(buffer.getvalue()))
    else:
        return '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,r.video_id,r.frame.frame_index)


def get_frame_url(r):
    return '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL,r.video_id,r.frame.frame_index)


def get_sequence_name(i,r):
    return "Indexer {} -> {} {} retriever".format(i.name,r.get_algorithm_display(),r.name)


def get_result_json(r):
    return dict(url=get_url(r), result_type="Region" if r.detection_id else "Frame", rank=r.rank, frame_id=r.frame_id,
                frame_index=r.frame.frame_index, distance=r.distance, video_id=r.video_id, video_name=r.video.name)


def get_query_region_json(rd):
    return dict(object_name=rd.object_name, event_id=rd.event_id, pk=rd.pk, x=rd.x, y=rd.y, w=rd.w,
                confidence=round(rd.confidence,2), text=rd.text, metadata=rd.metadata,
                region_type=rd.get_region_type_display(), h=rd.h, results=defaultdict(list))


def get_retrieval_event_name(r,rids_to_names):
    if r.retrieval_event_id not in rids_to_names:
        retriever = Retriever.objects.get(pk=r.retrieval_event.arguments['retriever_pk'])
        indexer = DeepModel.objects.get(name=r.retrieval_event.parent.arguments['index'],model_type=DeepModel.INDEXER)
        rids_to_names[r.retrieval_event_id] = get_sequence_name(indexer, retriever)
    return rids_to_names[r.retrieval_event_id]


# if request.method == 'POST':
#     form = UploadFileForm(request.POST, request.FILES)
#     name = form.cleaned_data['name']
#     rate = form.cleaned_data['nth'],
#     rescale = form.cleaned_data['rescale'] if 'rescale' in form.cleaned_data else 0
#     user = request.user if request.user.is_authenticated else None
#     if form.is_valid():
#         f = request.FILES['file']
#         uuid_fname = uuid.uuid1().__str__().replace('-', '_') + f.name.split('.')[-1]
#         with open('{}/inget/{}.{}'.format(settings.MEDIA_ROOT, uuid_fname),
#                   'wb+') as destination:
#             for chunk in f.chunks():
#                 destination.write(chunk)
#         handle_uploaded_file(request.FILES['file'], )
#         return redirect('video_list')
#     else:
#         raise ValueError
# else:
#     form = UploadFileForm()
#
#
#
# def handle_uploaded_file(filename, name, extract=True, user=None, rate=None, rescale=None):
#     create = []
#     process_spec = {'process_type': DVAPQL.PROCESS,'create':create}
#     if rate is None:
#         rate = defaults.DEFAULT_RATE
#     if rescale is None:
#         rescale = defaults.DEFAULT_RESCALE
#     filename = filename.lower()
#     if filename.endswith('.dva_export.zip'):
#         create.append({'MODEL': 'Video', 'spec': {'uploader_id': user.pk if user else None,'name': name},
#                        'tasks': [{'arguments': {'source':'INGEST','filename':filename},
#                                   'video_id': '__pk__',
#                                   'operation': 'perform_import',}
#                            ,]
#                        })
#     elif filename.endswith('.mp4') or filename.endswith('.zip'):
#         if extract:
#             if filename.endswith('.zip'):
#                 tasks=[{ 'arguments':{'rescale': rescale,
#                                       'next_tasks':defaults.DEFAULT_PROCESSING_PLAN_DATASET},
#                                       'operation': 'perform_dataset_extraction',
#                          }]
#             else:
#                 tasks = [{'arguments':{'next_tasks':[
#                                              {'operation':'perform_video_decode',
#                                                'arguments':{
#                                                    'segments_batch_size': defaults.DEFAULT_SEGMENTS_BATCH_SIZE,
#                                                    'rate': rate,
#                                                    'rescale': rescale,
#                                                    'next_tasks':defaults.DEFAULT_PROCESSING_PLAN_VIDEO
#                                                }
#                                               }
#                                             ]},
#                             'operation': 'perform_video_segmentation',
#                         }
#                     ]
#             create.append({'MODEL': 'Video', 'spec': {'uploader_id': user.pk if user else None,'name': name,
#                                                       'dataset':True if filename.endswith('zip') else False},
#                            'tasks': [{'arguments': {'source':'INGEST','filename':filename},
#                                       'video_id': '__pk__',
#                                       'operation': 'perform_import',}
#                                ,]
#                            })
#     else:
#         raise ValueError, "Extension {} not allowed".format(filename.split('.')[-1])
#     p = DVAPQL()
#     p.create_from_json(j=process_spec, user=user)
#     p.launch()
#
#     return video
