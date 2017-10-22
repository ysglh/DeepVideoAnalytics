import os, json, requests, copy, time, subprocess, logging, shutil, zipfile, boto3, random, calendar, shlex, sys, tempfile
from models import Video, QueryRegion, QueryRegionIndexVector, DVAPQL, Region, Frame, Segment, IndexEntries
from django.conf import settings
from PIL import Image
from . import serializers
from botocore.exceptions import ClientError

if settings.MEDIA_BUCKET:
    S3 = boto3.resource('s3')
    BUCKET = S3.Bucket(settings.MEDIA_BUCKET)
else:
    S3 = None
    BUCKET = None

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


def build_queryset(args,video_id=None,query_id=None):
    target = args['target']
    kwargs = args.get('filters',{})
    if video_id:
        kwargs['video_id'] = video_id
    if target == 'frames':
        queryset = Frame.objects.all().filter(**kwargs)
    elif target == 'regions':
        queryset = Region.objects.all().filter(**kwargs)
    elif target == 'query':
        kwargs['pk'] = query_id
        queryset = DVAPQL.objects.all().filter(**kwargs)
    elif target == 'query_regions':
        queryset = QueryRegion.objects.all().filter(**kwargs)
    elif target == 'query_region_index_vectors':
        queryset = QueryRegionIndexVector.objects.all().filter(**kwargs)
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


def download_and_get_query_path(start):
    local_path = "{}/queries/{}_{}.png".format(settings.MEDIA_ROOT, start.pk, start.parent_process_id)
    with open(local_path, 'w') as fh:
        fh.write(str(start.parent_process.image_data))
    return local_path


def download_and_get_query_region_path(start,regions):
    query_local_path = "{}/queries/{}_{}.png".format(settings.MEDIA_ROOT, start.pk, start.parent_process_id)
    if not os.path.isfile(query_local_path):
        with open(query_local_path, 'w') as fh:
            fh.write(str(start.parent_process.image_data))
    imdata = Image.open(query_local_path)
    rpaths = []
    for r in regions:
        region_path = "{}/queries/region_{}_{}.png".format(settings.MEDIA_ROOT, r.pk, start.parent_process_id)
        img2 = imdata.crop((r.x, r.y, r.x + r.w, r.y + r.h))
        img2.save(region_path)
        rpaths.append(region_path)
    return rpaths


def get_query_dimensions(start):
    query_local_path = "{}/queries/{}_{}.png".format(settings.MEDIA_ROOT, start.pk, start.parent_process_id)
    if not os.path.isfile(query_local_path):
        with open(query_local_path, 'w') as fh:
            fh.write(str(start.parent_process.image_data))
    imdata = Image.open(query_local_path)
    width, height = imdata.size
    return width, height


def download_model(root_dir, model_type_dir_name, dm):
    """
    Download model to filesystem
    """
    model_type_dir = "{}/{}/".format(root_dir, model_type_dir_name)
    if not os.path.isdir(model_type_dir):
        os.mkdir(model_type_dir)
    model_dir = "{}/{}/{}".format(root_dir, model_type_dir_name, dm.pk)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        if sys.platform == 'darwin':
            p = subprocess.Popen(['cp','/users/aub3/Dropbox/DeepVideoAnalytics/shared/{}'.format(dm.model_filename),'.'],cwd=model_dir)
            p.wait()
        else:
            p = subprocess.Popen(['wget','--quiet',dm.url],cwd=model_dir)
            p.wait()
        if dm.additional_files:
            for m in dm.additional_files:
                url = m['url']
                filename = m['filename']
                if sys.platform == 'darwin':
                    p = subprocess.Popen(
                        ['cp', '/users/aub3/Dropbox/DeepVideoAnalytics/shared/{}'.format(filename), '.'],cwd=model_dir)
                    p.wait()
                else:
                    p = subprocess.Popen(['wget', '--quiet', url], cwd=model_dir)
                    p.wait()


def crop_and_get_region_path(df,images,temp_root):
    if not df.materialized:
        frame_path = df.frame_path()
        if frame_path not in images:
            images[frame_path] = Image.open(frame_path)
        img2 = images[frame_path].crop((df.x, df.y, df.x + df.w, df.y + df.h))
        region_path = df.path(temp_root=temp_root)
        img2.save(region_path)
    else:
        return df.path()
    return region_path


def get_sync_paths(dirname,task_id):
    if dirname == 'indexes':
        f = [k.entries_path(media_root="") for k in IndexEntries.objects.filter(event_id=task_id)]
        f += [k.npy_path(media_root="") for k in IndexEntries.objects.filter(event_id=task_id)]
    elif dirname == 'frames':
        f = [k.path(media_root="") for k in Frame.objects.filter(event_id=task_id)]
    elif dirname == 'segments':
        f = [k.path(media_root="") for k in Segment.objects.filter(event_id=task_id)]
    elif dirname == 'regions':
        f = [k.path(media_root="") for k in Region.objects.filter(event_id=task_id)]
    else:
        raise NotImplementedError,"dirname : {} not configured".format(dirname)
    return f


def upload_file_to_remote(fpath):
    with open('{}{}'.format(settings.MEDIA_ROOT,fpath),'rb') as body:
        BUCKET.put_object(Key=fpath, Body=body)