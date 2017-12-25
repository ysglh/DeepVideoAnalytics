#!/usr/bin/env python
import django, sys, os, glob
sys.path.append('../server/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaui.view_shared import handle_uploaded_file

if __name__ == '__main__':
    for fname in glob.glob('ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/mp4")
        _ = handle_uploaded_file(f, name)
        break
    for fname in glob.glob('ci/example*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        _ = handle_uploaded_file(f, name)
        break
