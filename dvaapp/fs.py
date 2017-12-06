from django.conf import settings
import os
import uuid
import boto3
import shutil
import errno
import logging, subprocess, requests, zipfile
try:
    from google.cloud import storage
except:
    pass
if settings.MEDIA_BUCKET and settings.CLOUD_FS_PREFIX == 's3':
    S3_MODE = True
    GS_MODE = False
    S3 = boto3.resource('s3')
    BUCKET = S3.Bucket(settings.MEDIA_BUCKET)
elif settings.MEDIA_BUCKET and settings.CLOUD_FS_PREFIX == 'gs':
    S3_MODE = False
    GS_MODE = True
    GS = storage.Client()
    BUCKET = GS.get_bucket(settings.MEDIA_BUCKET)
else:
    S3_MODE = False
    GS_MODE = False
    S3 = None
    BUCKET = None


def mkdir_safe(dlpath):
    try:
        os.makedirs(os.path.dirname(dlpath))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def ingest_remote(dv,remote_path):
    logging.info("processing key {} ".format(remote_path))
    extension = remote_path.split('.')[-1]
    if remote_path.strip() and '.' in remote_path:
        fname = "{}/ingest/{}.{}".format(settings.MEDIA_ROOT, str(uuid.uuid1()).replace('-','_'), extension)
        get_remote_path_to_file(remote_path,fname)
        if not dv.name:
            dv.name = remote_path
        if remote_path.endswith('.dva_export.zip'):
            dv.create_directory(create_subdirs=False)
            os.rename(fname, '{}/{}/{}.{}'.format(settings.MEDIA_ROOT, dv.pk, dv.pk, extension))
            dv.uploaded = True
        elif remote_path.endswith('.mp4') or remote_path.endswith('.zip'):
            dv.create_directory(create_subdirs=True)
            os.rename(fname, '{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, dv.pk, dv.pk, extension))
            dv.uploaded = True
            if remote_path.endswith('.zip'):
                dv.dataset = True
        elif remote_path.endswith('json') or remote_path.endswith('.gz'):
            dv.create_directory(create_subdirs=True)
            os.rename(fname, '{}/{}/framelist.{}'.format(settings.MEDIA_ROOT, dv.pk, dv.pk, extension))
            dv.dataset = True
        else:
            raise ValueError, "Extension {} not allowed".format(remote_path)
        dv.save()
    else:
        # dv.create_directory(create_subdirs=False)
        # video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        # path = "{}/{}/".format(settings.MEDIA_ROOT, dv.pk)
        # download_s3_dir(key, path, bucket)
        # for filename in os.listdir(os.path.join(path, key)):
        #     shutil.move(os.path.join(path, key, filename), os.path.join(path, filename))
        # os.rmdir(os.path.join(path, key))
        # with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, dv.pk)) as input_json:
        #     video_json = json.load(input_json)
        # importer = serializers.VideoImporter(video=dv, json=video_json, root_dir=video_root_dir)
        # importer.import_video()
        # dv.uploaded = True
        # dv.save()
        raise NotImplementedError("Ability to import S3/GCS directories disabled")


def ingest_path(dv,path):
    extension = path.split('.')[-1]
    if path.endswith('dva_export.zip'):
        dv.create_directory(create_subdirs=False)
        if settings.DISABLE_NFS:
            if S3_MODE:
                source = '{}/{}'.format(settings.MEDIA_BUCKET, path.strip('/'))
                dest = '{}/{}.{}'.format(dv.pk,dv.pk,extension)
                try:
                    BUCKET.Object(dest).copy({'Bucket': settings.MEDIA_BUCKET, 'Key': path.strip('/')})
                except:
                    raise ValueError("Could not copy from {} to {}".format(source,dest))
                S3.Object(settings.MEDIA_BUCKET, path.strip('/')).delete()
            elif GS_MODE:
                raise NotImplementedError
            else:
                raise ValueError("NFS disabled and unknown cloud storage prefix")
        else:
            shutil.move(os.path.join(settings.MEDIA_ROOT,path.strip('/')),
                        '{}/{}/{}.{}'.format(settings.MEDIA_ROOT,dv.pk,dv.pk,extension))
    else:
        dv.create_directory(create_subdirs=True)
        if settings.DISABLE_NFS:
            if S3_MODE:
                source = '{}/{}'.format(settings.MEDIA_BUCKET, path.strip('/'))
                dest = '{}/video/{}.{}'.format(dv.pk,dv.pk,extension)
                try:
                    BUCKET.Object(dest).copy({'Bucket': settings.MEDIA_BUCKET, 'Key': path.strip('/')})
                except:
                    raise ValueError("Could not copy from {} to {}".format(source, dest))
                S3.Object(settings.MEDIA_BUCKET, path.strip('/')).delete()
            elif GS_MODE:
                raise NotImplementedError
            else:
                raise ValueError("NFS disabled and unknown cloud storage prefix")
        else:
            shutil.move(os.path.join(settings.MEDIA_ROOT,path.strip('/')),
                        '{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT,dv.pk,dv.pk,extension))


def ensure(path, dirnames=None, media_root=None):
    if BUCKET is not None:
        if media_root is None:
            media_root = settings.MEDIA_ROOT
        if dirnames is None:
            dirnames = {}
        if path.startswith('/') or media_root.endswith('/'):
            dlpath = "{}{}".format(media_root,path)
        else:
            dlpath = "{}/{}".format(media_root, path)
        dirname = os.path.dirname(dlpath)
        if os.path.isfile(dlpath):
            return True
        else:
            if dirname not in dirnames and not os.path.exists(dirname):
                mkdir_safe(dlpath)
            try:
                if S3_MODE:
                    BUCKET.download_file(path.strip('/'),dlpath)
                else:
                    with open(dlpath) as fout:
                        BUCKET.get_blob(path.strip('/')).download_to_file(fout)
            except:
                raise ValueError("path:{} dlpath:{}".format(path,dlpath))


def get_remote_path_to_file(remote_path,local_path):
    """
    # resource.meta.client.download_file(bucket, key, ofname, ExtraArgs={'RequestPayer': 'requester'})
    :param remote_path: e.g. s3://bucket/asd/asdsad/key.zip or gs:/bucket_name/key ..
    :param local_path:
    :return:
    """
    fs_type = remote_path[:2]
    bucket_name = remote_path[5:].split('/')[0]
    key = '/'.join(remote_path[5:].split('/')[1:])
    if fs_type == 's3':
        remote_bucket = S3.Bucket(bucket_name)
        remote_bucket.download_file(key, local_path)
    elif remote_path.starswith('gs'):
        remote_bucket = GS.get_bucket(settings.MEDIA_BUCKET)
        with open(local_path) as fout:
            remote_bucket.get_blob(key).download_to_file(fout)
    else:
        raise NotImplementedError("Unknown remote file system {}".format(remote_path))


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
        raise NotImplementedError("Google Storage CLI sync directory not implemented")
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


def download_yolo_detector(dm,path):
    dm.create_directory(create_subdirs=False)
    if 'www.dropbox.com' in path and not path.endswith('?dl=1'):
        r = requests.get(path + '?dl=1')
    else:
        r = requests.get(path)
    output_filename = "{}/models/{}.zip".format(settings.MEDIA_ROOT, dm.pk)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()
    source_zip = "{}/models/{}.zip".format(settings.MEDIA_ROOT, dm.pk)
    zipf = zipfile.ZipFile(source_zip, 'r')
    zipf.extractall("{}/models/{}/".format(settings.MEDIA_ROOT, dm.pk))
    zipf.close()
    os.remove(source_zip)