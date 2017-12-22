from django.conf import settings
import os
import shlex
import uuid
import boto3
import shutil
import errno
import logging, subprocess, requests, zipfile, urlparse
try:
    from google.cloud import storage
except:
    pass
try:
    S3 = boto3.resource('s3')
except:
    pass
try:
    GS = storage.Client()
except:
    pass
if settings.MEDIA_BUCKET and settings.CLOUD_FS_PREFIX == 's3':
    S3_MODE = True
    GS_MODE = False
    BUCKET = S3.Bucket(settings.MEDIA_BUCKET)
elif settings.MEDIA_BUCKET and settings.CLOUD_FS_PREFIX == 'gs':
    S3_MODE = False
    GS_MODE = True
    BUCKET = GS.get_bucket(settings.MEDIA_BUCKET)
else:
    S3_MODE = False
    GS_MODE = False
    BUCKET = None


def mkdir_safe(dlpath):
    try:
        os.makedirs(os.path.dirname(dlpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def retrieve_video_via_url(dv,url):
    dv.create_directory(create_subdirs=True)
    output_dir = "{}/{}/{}/".format(settings.MEDIA_ROOT, dv.pk, 'video')
    command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(url,dv.pk)
    logging.info(command)
    download = subprocess.Popen(shlex.split(command), cwd=output_dir)
    download.wait()
    if download.returncode != 0:
        raise ValueError,"Could not download the video"


def copy_remote(dv,path):
    extension = path.split('.')[-1]
    source = '{}/{}'.format(settings.MEDIA_BUCKET, path.strip('/'))
    dest = '{}/video/{}.{}'.format(dv.pk,dv.pk,extension)
    dv.create_directory() # for compatibility and to ensure that it sync does not fails.
    if S3_MODE:
        try:
            BUCKET.Object(dest).copy({'Bucket': settings.MEDIA_BUCKET, 'Key': path.strip('/')})
        except:
            raise ValueError("Could not copy from {} to {}".format(source,dest))
        S3.Object(settings.MEDIA_BUCKET, path.strip('/')).delete()
    elif GS_MODE:
        BUCKET.copy_blob(BUCKET.get_blob(path.strip('/')),BUCKET,new_name=dest)
        BUCKET.delete_blob(path.strip('/'))
    else:
        raise ValueError("NFS disabled and unknown cloud storage prefix")


def ensure(path, dirnames=None, media_root=None, safe=False, event_id=None):
    original_path = None
    if BUCKET is not None:
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        if dirnames is None:
            dirnames = {}
        if path.startswith('/') or media_root.endswith('/'):
            dlpath = "{}{}".format(media_root,path)
        else:
            dlpath = "{}/{}".format(media_root, path)
        if safe:
            if not event_id is None:
                original_path = dlpath
                dlpath = "{}.{}".format(dlpath,event_id)
            else:
                raise ValueError("Safe ensure must be used with event id instead got {}".format(event_id))
        dirname = os.path.dirname(dlpath)
        if os.path.isfile(dlpath):
            return True
        else:
            if dirname not in dirnames and not os.path.exists(dirname):
                mkdir_safe(dlpath)
            src = path.strip('/')
            if S3_MODE:
                try:
                    BUCKET.download_file(src,dlpath)
                except:
                    raise ValueError("{} to {}".format(path, dlpath))
            else:
                with open(dlpath,'w') as fout:
                    BUCKET.get_blob(src).download_to_file(fout)
        if safe:
            os.rename(dlpath,original_path)


def get_path_to_file(path,local_path):
    """
    # resource.meta.client.download_file(bucket, key, ofname, ExtraArgs={'RequestPayer': 'requester'})
    :param remote_path: e.g. s3://bucket/asd/asdsad/key.zip or gs:/bucket_name/key .. or /
    :param local_path:
    :return:
    """
    if settings.DISABLE_NFS and path.startswith('/ingest/'):
        if S3_MODE:
            path = "s3://{}{}".format(settings.MEDIA_BUCKET,path)
        elif GS_MODE:
            path = "gs://{}{}".format(settings.MEDIA_BUCKET,path)
        else:
            raise ValueError("NFS disabled but neither GS or S3 enabled.")
    fs_type = path[:2]
    if path.startswith('/ingest') and '..' not in path: # avoid relative imports outside media root
        shutil.move(os.path.join(settings.MEDIA_ROOT, path.strip('/')),local_path)
    elif path.startswith('http'):
        u = urlparse.urlparse(path)
        if u.hostname == 'www.dropbox.com' and not path.endswith('?dl=1'):
            r = requests.get(path + '?dl=1')
        else:
            r = requests.get(path)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        r.close()
    elif fs_type == 's3' and not path.endswith('/'):
        bucket_name = path[5:].split('/')[0]
        key = '/'.join(path[5:].split('/')[1:])
        remote_bucket = S3.Bucket(bucket_name)
        remote_bucket.download_file(key, local_path)
    elif path.startswith('gs') and not path.endswith('/'):
        bucket_name = path[5:].split('/')[0]
        key = '/'.join(path[5:].split('/')[1:])
        remote_bucket = GS.get_bucket(bucket_name)
        with open(local_path,'w') as fout:
            remote_bucket.get_blob(key).download_to_file(fout)
    else:
        raise NotImplementedError("importing S3/GCS directories disabled or Unknown file system {}".format(path))


def upload_file_to_remote(fpath):
    if S3_MODE:
        with open('{}{}'.format(settings.MEDIA_ROOT,fpath),'rb') as body:
            S3.Object(settings.MEDIA_BUCKET,fpath.strip('/')).put(Body=body)
    else:
        fblob = BUCKET.blob(fpath.strip('/'))
        fblob.upload_from_filename(filename='{}{}'.format(settings.MEDIA_ROOT,fpath))


def download_video_from_remote_to_local(dv):
    logging.info("Syncing entire directory for {}".format(dv.pk))
    if S3_MODE:
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
    else:
        raise NotImplementedError


def upload_video_to_remote(video_id):
    logging.info("Syncing entire directory for {}".format(video_id))
    src = '{}/{}/'.format(settings.MEDIA_ROOT, video_id)
    if S3_MODE:
        dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, video_id)
        command = " ".join(['aws', 's3', 'sync', '--quiet', src, dest])
        syncer = subprocess.Popen(['aws', 's3', 'sync', '--quiet', '--size-only', src, dest])
        syncer.wait()
        if syncer.returncode != 0:
            raise ValueError, "Error while executing : {}".format(command)
    elif GS_MODE:
        root_length = len(settings.MEDIA_ROOT)
        for root, directories, filenames in os.walk(src):
            for filename in filenames:
                path = os.path.join(root,filename)
                upload_file_to_remote(path[root_length:])
    else:
        raise ValueError


def download_s3_dir(dist, local, bucket, client = None, resource = None):
    """
    Taken from http://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
    :param client:
    :param resource:
    :param dist:
    :param local:
    :param bucket:
    :return:
    """
    if client is None and resource is None:
        client = boto3.client('s3')
        resource = boto3.resource('s3')
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist, RequestPayer='requester'):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_s3_dir(subdir.get('Prefix'), local, bucket, client, resource)
        if result.get('Contents') is not None:
            for ffile in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + ffile.get('Key'))):
                    os.makedirs(os.path.dirname(local + os.sep + ffile.get('Key')))
                resource.meta.client.download_file(bucket, ffile.get('Key'), local + os.sep + ffile.get('Key'),
                                                   ExtraArgs={'RequestPayer': 'requester'})


