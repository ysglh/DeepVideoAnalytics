import django,os,glob
from django.core.files.uploadedfile import SimpleUploadedFile


if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.tasks import perform_detection
    perform_detection(2)
