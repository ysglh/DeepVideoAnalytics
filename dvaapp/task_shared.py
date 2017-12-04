import os, json, requests, copy, time, subprocess, logging, shutil, zipfile, random, calendar, shlex, sys, tempfile
from models import Video, QueryRegion, QueryRegionIndexVector, DVAPQL, Region, Frame, Segment, IndexEntries, TEvent
from django.conf import settings
from PIL import Image
from . import serializers
from .fs import ensure, upload_file_to_remote, upload_video_to_remote, download_s3_dir


def pid_exists(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def relaunch_failed_task(old, app):
    """
    TODO: Relaunch failed tasks, requires a rethink in how we store number of attempts.
    Cleanup of objects created by previous task that that failed.
    :param old:
    :param app:
    :return:
    """
    if old.errored:
        next_task = TEvent.objects.create(video=old.video, operation=old.operation, arguments=old.arguments,
                                          parent=old.parent, parent_process=old.parent_process, queue=old.queue)
        app.send_task(next_task.operation, args=[next_task.pk, ], queue=old.queue)
    else:
        raise ValueError("Task not errored")


def launch_worker(queue_name, worker_name):
    p = subprocess.Popen(['fab', 'startq:{}'.format(queue_name)], close_fds=True)
    message = "launched {} with pid {} on {}".format(queue_name, p.pid, worker_name)
    return message


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


def perform_s3_export(dv,path,export_event_pk=None):
    cwd_path = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
    a = serializers.VideoExportSerializer(instance=dv)
    data = copy.deepcopy(a.data)
    data['labels'] = serializers.serialize_video_labels(dv)
    if export_event_pk:
        data['export_event_pk'] = export_event_pk
    with file("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk), 'w') as output:
        json.dump(data, output)
    if path.startswith('s3://'):
        upload = subprocess.Popen(args=["aws", "s3", "sync",'--quiet', ".",path],cwd=cwd_path)
        upload.communicate()
        upload.wait()
        return upload.returncode
    elif path.startswith('gs://'):
        raise NotImplementedError


def import_remote(start,dv):
    remote_path = start.arguments['path']
    logging.info("processing key {} ".format(remote_path))
    if dv is None:
        dv = Video()
        dv.name = "pending {}".format(remote_path)
        dv.save()
        start.video = dv
        start.save()
    path = "{}/{}/".format(settings.MEDIA_ROOT, start.video.pk)
    if remote_path.strip() and (remote_path.endswith('.zip') or remote_path.endswith('.mp4')):
        fname = 'temp_' + str(time.time()).replace('.', '_') + '_' + str(random.randint(0, 100)) + '.' + \
                remote_path.split('.')[-1]
        command = ["aws", "s3", "cp", '--quiet', remote_path, fname]
        path = "{}/".format(settings.MEDIA_ROOT)
        download = subprocess.Popen(args=command, cwd=path)
        download.communicate()
        download.wait()
        if download.returncode != 0:
            start.errored = True
            start.error_message = "return code for '{}' was {}".format(" ".join(command), download.returncode)
            start.save()
            raise ValueError, start.error_message
        handle_downloaded_file("{}/{}".format(settings.MEDIA_ROOT, fname), start.video, remote_path)
    else:
        start.video.create_directory(create_subdirs=False)
        command = ["aws", "s3", "cp", '--quiet', remote_path, '.', '--recursive']
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


def import_url(dv,download_url):
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
    if settings.DISABLE_NFS:
        fname = "/{}/{}.zip".format(video_id, video_id)
        logging.info("Downloading {}".format(fname))
        ensure(fname)
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
    if key.endswith('.dva_export.zip'):
        ofname = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
        raise NotImplementedError
        # resource.meta.client.download_file(bucket, key, ofname, ExtraArgs={'RequestPayer': 'requester'})
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
        download_s3_dir(key, path, bucket)
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
    # if NFS is disabled upload to the bucket
    if settings.DISABLE_NFS:
        upload_file_to_remote("/exports/{}".format(file_name))
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
    elif target == 'segments':
        queryset = Segment.objects.filter(**kwargs)
    else:
        raise ValueError("target {} not found".format(target))
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
        if settings.DEV_ENV:
            p = subprocess.Popen(['cp','/users/aub3/shared/{}'.format(dm.model_filename),'.'],cwd=model_dir)
            p.wait()
        else:
            p = subprocess.Popen(['wget','--quiet',dm.url],cwd=model_dir)
            p.wait()
        if dm.additional_files:
            for m in dm.additional_files:
                url = m['url']
                filename = m['filename']
                if settings.DEV_ENV:
                    p = subprocess.Popen(['cp', '/users/aub3/shared/{}'.format(filename), '.'],cwd=model_dir)
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


def ensure_files(queryset, target):
    dirnames = {}
    if target == 'frames':
        for k in queryset:
            ensure(k.path(media_root=''),dirnames)
    elif target == 'regions':
        for k in queryset:
            if k.materialized:
                ensure(k.path(media_root=''), dirnames)
            else:
                ensure(k.frame_path(media_root=''), dirnames)
    elif target == 'segments':
        for k in queryset:
            ensure(k.path(media_root=''),dirnames)
            ensure(k.framelist_path(media_root=''), dirnames)
    elif target == 'indexes':
        for k in queryset:
            ensure(k.npy_path(media_root=''), dirnames)
            ensure(k.entries_path(media_root=''), dirnames)
    else:
        raise NotImplementedError


def import_frame_regions_json(regions_json,video,event_id):
    """
    Import regions from a JSON with frames identified by immuntable identifiers such as filename/path
    :param regions_json:
    :param video:
    :param event_id:
    :return:
    """
    video_id = video.pk
    filename_to_pk = {}
    frame_index_to_pk = {}
    if video.dataset:
        # For dataset frames are identified by subdir/filename
        filename_to_pk = { df.original_path(): (df.pk, df.frame_index)
                           for df in Frame.objects.filter(video_id=video_id)}
    else:
        # For videos frames are identified by frame index
        frame_index_to_pk = { df.frame_index: (df.pk, df.segment_index) for df in
                        Frame.objects.filter(video_id=video_id)}
    regions = []
    for k in regions_json:
        r = Region()
        if k['target'] == 'filename':
            fname = k['filename']
            if not fname.startswith('/'):
                fname = '/{}'.format(fname)
            pk,findx = filename_to_pk[fname]
            r.frame_id = pk
            r.frame_index = findx
        elif k['target'] == 'index':
            pk,sindx = frame_index_to_pk[k['frame_index']]
            r.frame_id = pk
            r.frame_index = k['frame_index']
            r.segment_index = sindx
        else:
            raise ValueError('invalid target: {}'.format(k['target']))
        r.video_id = video_id
        r.event_id = event_id
        r.region_type = k['region_type']
        r.materialized = k.get('materialized',False)
        r.full_frame = k.get('full_frame',False)
        r.x = k['x']
        r.y = k['y']
        r.w = k['w']
        r.h = k['h']
        r.metadata = k['metadata']
        r.text = k['text']
    Region.objects.bulk_create(regions,1000)


def get_sync_paths(dirname,task_id):
    if dirname == 'indexes':
        f = [k.entries_path(media_root="") for k in IndexEntries.objects.filter(event_id=task_id)]
        f += [k.npy_path(media_root="") for k in IndexEntries.objects.filter(event_id=task_id)]
    elif dirname == 'frames':
        f = [k.path(media_root="") for k in Frame.objects.filter(event_id=task_id)]
    elif dirname == 'segments':
        f = []
        for k in Segment.objects.filter(event_id=task_id):
            f.append(k.path(media_root=""))
            f.append(k.framelist_path(media_root=""))
    elif dirname == 'regions':
        f = [k.path(media_root="") for k in Region.objects.filter(event_id=task_id) if k.materialized]
    else:
        raise NotImplementedError,"dirname : {} not configured".format(dirname)
    return f


def upload(dirname,event_id,video_id):
    if dirname:
        fnames = get_sync_paths(dirname, event_id)
        logging.info("Syncing {} containing {} files".format(dirname, len(fnames)))
        for fp in fnames:
            upload_file_to_remote(fp)
    else:
        upload_video_to_remote(video_id)
