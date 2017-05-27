import os,json,requests,base64
from models import Video,TEvent,AppliedLabel,Region,Frame,VDNDataset,VDNServer,Query
from django.conf import settings
from dva.celery import app
from celery.result import AsyncResult
import boto3


def refresh_task_status():
    for t in TEvent.objects.all().filter(started=True,completed=False,errored=False):
        if AsyncResult(t.id).status == 'FAILURE':
            t.errored = True
            t.save()


def create_video_folders(video,create_subdirs=True):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
    if create_subdirs:
        os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/detections/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, video.pk))


def handle_uploaded_file(f,name,extract=True,user=None,perform_scene_detection=True,rate=30,rescale=0):
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
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        with open('{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]), 'wb+') as destination:
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
            app.send_task(name=task_name, args=[extract_frames_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise ValueError,"Extension {} not allowed".format(filename.split('.')[-1])
    return video


def handle_downloaded_file(downloaded,video,name,extract=True,user=None,perform_scene_detection=True,rate=30,rescale=0,):
    video.name = name
    video.save()
    filename = downloaded.split('/')[-1]
    if filename.endswith('.dva_export.zip'):
        create_video_folders(video, create_subdirs=False)
        os.rename(downloaded,'{}/{}/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]))
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        os.rename(downloaded,'{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            extract_frames_task = TEvent()
            extract_frames_task.arguments_json = json.dumps({'perform_scene_detection': perform_scene_detection,'rate': rate,'rescale': rescale})
            extract_frames_task.video = video
            task_name = 'extract_frames_by_id'
            extract_frames_task.operation = task_name
            extract_frames_task.save()
            app.send_task(name=task_name, args=[extract_frames_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise ValueError,"Extension {} not allowed".format(filename.split('.')[-1])
    return video


def create_annotation(form,object_name,labels,frame):
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


def handle_youtube_video(name,url,extract=True,user=None,perform_scene_detection=True,rate=30,rescale=0):
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


def pull_vdn_dataset_list(pk):
    """
    Pull list of datasets from configured VDN servers
    """
    server = VDNServer.objects.get(pk=pk)
    r = requests.get("{}vdn/api/datasets/".format(server.url))
    response = r.json()
    datasets = []
    for d in response['results']:
        datasets.append(d)
    while response['next']:
        r = requests.get("{}vdn/api/datasets/".format(server))
        response = r.json()
        for d in response['results']:
            datasets.append(d)
    server.last_response_datasets = json.dumps(datasets)
    server.save()
    return server,datasets


def create_query(count,approximate,selected,excluded_pks,image_data_url):
    query = Query()
    query.count = count
    if excluded_pks:
        query.excluded_index_entries_pk = [int(k) for k in excluded_pks]
    query.selected_indexers = selected
    query.approximate = approximate
    image_data = base64.decodestring(image_data_url[22:])
    if settings.HEROKU_DEPLOY:
        query.image_data = image_data
    query.save()
    dv = Video()
    dv.name = 'query_{}'.format(query.pk)
    dv.dataset = True
    dv.query = True
    dv.parent_query = query
    dv.save()
    create_video_folders(dv)
    query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, query.pk)
    query_frame_path = "{}/{}/frames/0.png".format(settings.MEDIA_ROOT, dv.pk)
    if settings.HEROKU_DEPLOY:
        s3 = boto3.resource('s3')
        s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_path, Body=image_data)
        s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_frame_path, Body=image_data)
    else:
        with open(query_path, 'w') as fh:
            fh.write(image_data)
        with open(query_frame_path, 'w') as fh:
            fh.write(image_data)
    return query,dv


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


def import_vdn_dataset_url(server,url,user):
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
    primary_key = video.pk
    create_video_folders(video, create_subdirs=False)
    task_name = 'import_video_by_id'
    import_video_task = TEvent()
    import_video_task.video = video
    import_video_task.save()
    app.send_task(name=task_name, args=[import_video_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
