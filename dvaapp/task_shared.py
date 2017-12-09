import os, json, requests, copy, time, subprocess, logging, shutil, zipfile, uuid, calendar, shlex, sys, tempfile, uuid
from models import Video, QueryRegion, QueryRegionIndexVector, DVAPQL, Region, Frame, Segment, IndexEntries, TEvent
from django.conf import settings
from PIL import Image
from . import serializers
from .fs import ensure, upload_file_to_remote, upload_video_to_remote, get_path_to_file


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


def import_path(dv,path,export=False,framelist=False):
    if export:
        dv.create_directory(create_subdirs=False)
        output_filename = "{}/{}/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk)
    else:
        dv.create_directory(create_subdirs=True)
        extension = path.split('?')[0].split('.')[-1]
        if framelist:
            output_filename = "{}/{}/framelist.{}".format(settings.MEDIA_ROOT, dv.pk, extension)
        else:
            output_filename = "{}/{}/video/{}.{}".format(settings.MEDIA_ROOT, dv.pk, dv.pk, extension)
    get_path_to_file(path,output_filename)


def load_dva_export_file(dv):
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


def load_frame_list(dv,event_id):
    """
    Add ability load frames & regions specified in a JSON file and then automatically
    retrieve them in a distributed manner them through CPU workers.
    :param dv:
    :return:
    """
    frame_list = dv.get_frame_list()
    temp_path = "{}.jpg".format(uuid.uuid1()).replace('-', '_')
    frame_index = 0
    video_id = dv.pk
    frame_index_to_regions = {}
    frames = []
    for i, f in enumerate(frame_list['frames']):
        try:
            get_path_to_file(f['path'],temp_path)
        except:
            logging.exception("Failed to get {}".format(f['path']))
            pass
        else:
            df, drs = serializers.import_frame_json(f,frame_index,event_id,video_id)
            frame_index_to_regions[frame_index] = drs
            frames.append(df)
            shutil.move(temp_path,df.path())
            frame_index += 1
    fids = Frame.objects.bulk_create(frames,1000)
    regions = []
    for i,f in enumerate(fids):
        region_list = frame_index_to_regions[i]
        logging.info(region_list)
        logging.info(len(region_list))
        for dr in region_list:
            dr.frame_id = f.id
            regions.append(dr)
    Region.objects.bulk_create(regions,1000)
    dv.uploaded = True
    dv.frames = frame_index


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
        e = TEvent.objects.get(pk=task_id)
        if e.operation == 'perform_transformation': # TODO: transformation events merely materialize, fix this
            fargs = copy.deepcopy(e.arguments['filters'])
            fargs['materialized'] = True
            fargs['video_id'] = e.video_id
            f = [k.path(media_root="") for k in Region.objects.filter(**fargs)]
        else:
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
