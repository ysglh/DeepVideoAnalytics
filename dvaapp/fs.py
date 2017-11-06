from django.conf import settings
import os
import boto3
import errno

if settings.MEDIA_BUCKET:
    S3 = boto3.resource('s3')
    BUCKET = S3.Bucket(settings.MEDIA_BUCKET)
else:
    S3 = None
    BUCKET = None


def ensure(path,dirnames=None,media_root=None):
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
                try:
                    os.makedirs(os.path.dirname(dlpath))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                try:
                    BUCKET.download_file(path.strip('/'),dlpath)
                except:
                    raise ValueError("path:{} dlpath:{}".format(path,dlpath))


def upload_file_to_remote(fpath):
    with open('{}{}'.format(settings.MEDIA_ROOT,fpath),'rb') as body:
        S3.Object(settings.MEDIA_BUCKET,fpath.strip('/')).put(Body=body)
