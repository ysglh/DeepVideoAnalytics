from django.conf import settings
import os
import boto3
import errno
import logging, subprocess
from models import Frame, IndexEntries, Segment, Region

if settings.MEDIA_BUCKET:
    S3 = boto3.resource('s3')
    BUCKET = S3.Bucket(settings.MEDIA_BUCKET)
else:
    S3 = None
    BUCKET = None


def mkdir_safe(dlpath):
    try:
        os.makedirs(os.path.dirname(dlpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def ensure(path, dirnames=None, media_root=None):
    if BUCKET is not None:
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        if dirnames is None:
            dirnames = {}
        dlpath = "{}{}".format(media_root,path)
        dirname = os.path.dirname(dlpath)
        if os.path.isfile(dlpath):
            return True
        else:
            if dirname not in dirnames and not os.path.exists(dirname):
                mkdir_safe(dlpath)
            try:
                BUCKET.download_file(path.strip('/'),dlpath)
            except:
                raise ValueError("path:{} dlpath:{}".format(path,dlpath))


def upload_file_to_remote(fpath):
    with open('{}{}'.format(settings.MEDIA_ROOT,fpath),'rb') as body:
        S3.Object(settings.MEDIA_BUCKET,fpath.strip('/')).put(Body=body)


def download_video_from_remote_to_local(dv):
    logging.info("Syncing entire directory for {}".format(dv.pk))
    dest = '{}/{}/'.format(settings.MEDIA_ROOT, dv.pk)
    src = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, dv.pk)
    try:
        os.mkdir(dest)
    except:
        pass
    command = " ".join(['aws', 's3', 'sync', '--quiet', src, dest])
    syncer = subprocess.Popen(['aws', 's3', 'sync', '--quiet', '--size-only', src, dest])
    syncer.wait()
    if syncer.returncode != 0:
        raise ValueError, "Error while executing : {}".format(command)


def upload(dirname,event_id,video_id):
    if dirname:
        fnames = get_sync_paths(dirname, event_id)
        logging.info("Syncing {} containing {} files".format(dirname, len(fnames)))
        for fp in fnames:
            upload_file_to_remote(fp)
    else:
        logging.info("Syncing entire directory for {}".format(video_id))
        src = '{}/{}/'.format(settings.MEDIA_ROOT, video_id)
        dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, video_id)
        command = " ".join(['aws', 's3', 'sync', '--quiet', src, dest])
        syncer = subprocess.Popen(['aws', 's3', 'sync', '--quiet', '--size-only', src, dest])
        syncer.wait()
        if syncer.returncode != 0:
            raise ValueError,"Error while executing : {}".format(command)


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
