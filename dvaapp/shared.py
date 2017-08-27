import os, json, requests, copy, time, subprocess, logging, shutil, zipfile, boto3, random, calendar, shlex
from models import Video, TEvent,  Region, VDNServer, Detector, DeletedVideo, Label,\
    RegionLabel
from django.conf import settings
from django_celery_results.models import TaskResult
from collections import defaultdict
from operations import processing
from models import DVAPQL,Region,Frame
from . import serializers
from botocore.exceptions import ClientError
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


def retrieve_video_via_url(dv,media_dir):
    dv.create_directory(create_subdirs=True)
    output_dir = "{}/{}/{}/".format(media_dir, dv.pk, 'video')
    command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(dv.url,dv.pk)
    logging.info(command)
    download = subprocess.Popen(shlex.split(command), cwd=output_dir)
    download.wait()
    if download.returncode != 0:
        raise ValueError,"Could not download the video"


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
    detector = Detector()
    detector.name = response['name']
    detector.detector_type = response.get('detector_type',detector.YOLO)
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


def download_s3_dir(client, resource, dist, local, bucket):
    """
    Taken from http://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
    :param client:
    :param resource:
    :param dist:
    :param local:
    :param bucket:
    :return:
    """
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist, RequestPayer='requester'):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_s3_dir(client, resource, subdir.get('Prefix'), local, bucket)
        if result.get('Contents') is not None:
            for ffile in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + ffile.get('Key'))):
                    os.makedirs(os.path.dirname(local + os.sep + ffile.get('Key')))
                resource.meta.client.download_file(bucket, ffile.get('Key'), local + os.sep + ffile.get('Key'),
                                                   ExtraArgs={'RequestPayer': 'requester'})


def perform_s3_export(dv,s3key,s3bucket,s3region,export_event_pk=None,create_bucket=False):
    s3 = boto3.resource('s3')
    if create_bucket:
        if s3region == 'us-east-1':
            s3.create_bucket(Bucket=s3bucket)
        else:
            s3.create_bucket(Bucket=s3bucket, CreateBucketConfiguration={'LocationConstraint': s3region})
        time.sleep(20)  # wait for it to create the bucket
    path = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
    a = serializers.VideoExportSerializer(instance=dv)
    data = copy.deepcopy(a.data)
    data['labels'] = serializers.serialize_video_labels(dv)
    if export_event_pk:
        data['export_event_pk'] = export_event_pk
    exists = False
    try:
        s3.Object(s3bucket, '{}/table_data.json'.format(s3key).replace('//', '/')).load()
    except ClientError as e:
        if e.response['Error']['Code'] != "404":
            raise ValueError,"Key s3://{}/{}/table_data.json already exists".format(s3bucket,s3key)
    else:
        return -1, "Error key already exists"
    with file("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk), 'w') as output:
        json.dump(data, output)
    upload = subprocess.Popen(args=["aws", "s3", "sync",'--quiet', ".", "s3://{}/{}/".format(s3bucket,s3key)],cwd=path)
    upload.communicate()
    upload.wait()
    return upload.returncode


def celery_40_bug_hack(start):
    """
    Celery 4.0.2 retries tasks due to ACK issues when running in solo mode,
    Since Tensorflow ncessiates use of solo mode, we can manually check if the task is has already run and quickly finis it
    Since the code never uses Celery results except for querying and retries are handled at application level this solves the
    issue
    :param start:
    :return:
    """
    return start.started


def import_s3(start,dv):
    s3key = start.arguments['key']
    s3bucket = start.arguments['bucket']
    logging.info("processing key  {}space".format(s3key))
    if dv is None:
        dv = Video()
        dv.name = "pending S3 import from s3://{}/{}".format(s3bucket, s3key)
        dv.save()
        start.video = dv
        start.save()
    path = "{}/{}/".format(settings.MEDIA_ROOT, start.video.pk)
    if s3key.strip() and (s3key.endswith('.zip') or s3key.endswith('.mp4')):
        fname = 'temp_' + str(time.time()).replace('.', '_') + '_' + str(random.randint(0, 100)) + '.' + \
                s3key.split('.')[-1]
        command = ["aws", "s3", "cp", '--quiet', "s3://{}/{}".format(s3bucket, s3key), fname]
        path = "{}/".format(settings.MEDIA_ROOT)
        download = subprocess.Popen(args=command, cwd=path)
        download.communicate()
        download.wait()
        if download.returncode != 0:
            start.errored = True
            start.error_message = "return code for '{}' was {}".format(" ".join(command), download.returncode)
            start.save()
            raise ValueError, start.error_message
        handle_downloaded_file("{}/{}".format(settings.MEDIA_ROOT, fname), start.video,
                               "s3://{}/{}".format(s3bucket, s3key))
    else:
        start.video.create_directory(create_subdirs=False)
        command = ["aws", "s3", "cp", '--quiet', "s3://{}/{}/".format(s3bucket, s3key), '.', '--recursive']
        command_exec = " ".join(command)
        download = subprocess.Popen(args=command, cwd=path)
        download.communicate()
        download.wait()
        if download.returncode != 0:
            start.errored = True
            start.error_message = "return code for '{}' was {}".format(command_exec, download.returncode)
            start.save()
            raise ValueError, start.error_message
        with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, start.video.pk)) as input_json:
            video_json = json.load(input_json)
        importer = serializers.VideoImporter(video=start.video, json=video_json, root_dir=path)
        importer.import_video()


def import_vdn_url(dv,download_url):
    dv.create_directory(create_subdirs=False)
    if 'www.dropbox.com' in download_url and not download_url.endswith('?dl=1'):
        r = requests.get(download_url + '?dl=1')
    else:
        r = requests.get(download_url)
    output_filename = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()
    zipf = zipfile.ZipFile("{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk), 'r')
    zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, dv.pk))
    zipf.close()
    video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
    for k in os.listdir(video_root_dir):
        unzipped_dir = "{}{}".format(video_root_dir, k)
        if os.path.isdir(unzipped_dir):
            for subdir in os.listdir(unzipped_dir):
                shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
            shutil.rmtree(unzipped_dir)
            break
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk)) as input_json:
        video_json = json.load(input_json)
    importer = serializers.VideoImporter(video=dv, json=video_json, root_dir=video_root_dir)
    importer.import_video()
    source_zip = "{}/{}.zip".format(video_root_dir, dv.pk)
    os.remove(source_zip)
    dv.uploaded = True
    dv.save()


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


def import_vdn_s3(dv,key,bucket):
    dv.create_directory(create_subdirs=False)
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    if key.endswith('.dva_export.zip'):
        ofname = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
        resource.meta.client.download_file(bucket, key, ofname, ExtraArgs={'RequestPayer': 'requester'})
        zipf = zipfile.ZipFile(ofname, 'r')
        zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, dv.pk))
        zipf.close()
        video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        for k in os.listdir(video_root_dir):
            unzipped_dir = "{}{}".format(video_root_dir, k)
            if os.path.isdir(unzipped_dir):
                for subdir in os.listdir(unzipped_dir):
                    shutil.move("{}/{}".format(unzipped_dir, subdir), "{}".format(video_root_dir))
                shutil.rmtree(unzipped_dir)
                break
        source_zip = "{}/{}.zip".format(video_root_dir, dv.pk)
        os.remove(source_zip)
    else:
        video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        path = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        download_s3_dir(client, resource, key, path, bucket)
        for filename in os.listdir(os.path.join(path, key)):
            shutil.move(os.path.join(path, key, filename), os.path.join(path, filename))
        os.rmdir(os.path.join(path, key))
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk)) as input_json:
        video_json = json.load(input_json)
    importer = serializers.VideoImporter(video=dv, json=video_json, root_dir=video_root_dir)
    importer.import_video()
    dv.uploaded = True
    dv.save()


def export_file(video_obj,export_event_pk=None):
    video_id = video_obj.pk
    file_name = '{}_{}.dva_export.zip'.format(video_id, int(calendar.timegm(time.gmtime())))
    try:
        os.mkdir("{}/{}".format(settings.MEDIA_ROOT, 'exports'))
    except:
        pass
    outdirname = "{}/exports/{}".format(settings.MEDIA_ROOT, video_id)
    if os.path.isdir(outdirname):
        shutil.rmtree(outdirname)
    shutil.copytree('{}/{}'.format(settings.MEDIA_ROOT, video_id),
                    "{}/exports/{}".format(settings.MEDIA_ROOT, video_id))
    a = serializers.VideoExportSerializer(instance=video_obj)
    data = copy.deepcopy(a.data)
    data['labels'] = serializers.serialize_video_labels(video_obj)
    if export_event_pk:
        data['export_event_pk'] = export_event_pk
    with file("{}/exports/{}/table_data.json".format(settings.MEDIA_ROOT, video_id), 'w') as output:
        json.dump(data, output)
    zipper = subprocess.Popen(['zip', file_name, '-r', '{}'.format(video_id)],
                              cwd='{}/exports/'.format(settings.MEDIA_ROOT))
    zipper.wait()
    shutil.rmtree("{}/exports/{}".format(settings.MEDIA_ROOT, video_id))
    return file_name


def build_queryset(args,video_id=None):
    target = args['target']
    kwargs = args.get('filters',{})
    if video_id:
        kwargs['video_id'] = video_id
    if target == 'frames':
        queryset = Frame.objects.all().filter(**kwargs)
    elif target == 'regions':
        queryset = Region.objects.all().filter(**kwargs)
    else:
        queryset = None
        raise ValueError
    return queryset,target


def import_external(args):
    dataset = args['dataset']
    if dataset == 'YFCC':
        bucket = args.get('bucket','yahoo-webscope')
        prefix = args.get('prefix','I3set14')
        force_download = args.get('force_download', False)
        fnames = args.get('files',['yfcc100m_autotags.bz2','yfcc100m_dataset.bz2','yfcc100m_places.bz2'])
        for fname in fnames:
            outfile = "{}/external/{}".format(settings.MEDIA_ROOT,fname)
            if (not os.path.isfile(outfile)) or force_download:
                command = ['aws','s3','cp',
                           's3://{}/{}/{}'.format(bucket,prefix,fname),
                           outfile]
                sp = subprocess.Popen(command)
                sp.wait()
                if sp.returncode != 0:
                    raise ValueError,"Returncode {} for command {}".format(sp.returncode," ".join(command))
    elif dataset == 'AMOS':
        raise NotImplementedError
    else:
        raise ValueError,"dataset:{} not configured".format(dataset)