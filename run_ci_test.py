import django,os,glob
from django.core.files.uploadedfile import SimpleUploadedFile


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.views import handle_uploaded_file,handle_youtube_video
    from dvaapp.models import Video
    from dvaapp.tasks import extract_frames,perform_indexing,perform_detection
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f,name,False)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)
    handle_youtube_video('yolov2', 'https://www.youtube.com/watch?v=C4qgAaxB_pc')
    for v in Video.objects.all():
        extract_frames(v.pk)
        perform_indexing(v.pk)
        perform_detection(v.pk)