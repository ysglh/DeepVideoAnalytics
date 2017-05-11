import os,json
from models import Video,TEvent
from django.conf import settings
from dva.celery import app


def create_video_folders(video,create_subdirs=True):
    os.mkdir('{}/{}'.format(settings.MEDIA_ROOT, video.pk))
    if create_subdirs:
        os.mkdir('{}/{}/video/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/frames/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/indexes/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/detections/'.format(settings.MEDIA_ROOT, video.pk))
        os.mkdir('{}/{}/audio/'.format(settings.MEDIA_ROOT, video.pk))


def handle_uploaded_file(f,name,extract=True,user=None,perform_scene_detection=True,rate=30,rescale=0):
    video = Video()
    if user:
        video.uploader = user
    video.name = name
    video.save()
    primary_key = video.pk
    filename = f.name
    filename = filename.lower()
    if filename.endswith('.dva_export.zip'):
        create_video_folders(video, create_subdirs=False)
        with open('{}/{}/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        with open('{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            extract_frames_task = TEvent()
            extract_frames_task.arguments_json = json.dumps({'perform_scene_detection': perform_scene_detection,
                                                             'rate': rate,
                                                             'rescale': rescale})
            extract_frames_task.video = video
            task_name = 'extract_frames_by_id'
            extract_frames_task.operation = task_name
            extract_frames_task.save()
            app.send_task(name=task_name, args=[extract_frames_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise ValueError,"Extension {} not allowed".format(filename.split('.')[-1])
    return video


def handle_downloaded_file(downloaded,video,name,extract=True,user=None,perform_scene_detection=True,rate=30,rescale=0,):
    video.name = name
    video.save()
    filename = downloaded.split('/')[-1]
    if filename.endswith('.dva_export.zip'):
        os.rename(downloaded,'{}/{}/{}.{}'.format(settings.MEDIA_ROOT,video.pk,video.pk,filename.split('.')[-1]))
        video.uploaded = True
        video.save()
        task_name = 'import_video_by_id'
        import_video_task = TEvent()
        import_video_task.video = video
        import_video_task.save()
        app.send_task(name=task_name, args=[import_video_task.pk,], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    elif filename.endswith('.mp4') or filename.endswith('.flv') or filename.endswith('.zip'):
        create_video_folders(video, create_subdirs=True)
        os.rename(downloaded,'{}/{}/video/{}.{}'.format(settings.MEDIA_ROOT, video.pk, video.pk, filename.split('.')[-1]))
        video.uploaded = True
        if filename.endswith('.zip'):
            video.dataset = True
        video.save()
        if extract:
            extract_frames_task = TEvent()
            extract_frames_task.arguments_json = json.dumps({'perform_scene_detection': perform_scene_detection,'rate': rate,'rescale': rescale})
            extract_frames_task.video = video
            task_name = 'extract_frames_by_id'
            extract_frames_task.operation = task_name
            extract_frames_task.save()
            app.send_task(name=task_name, args=[extract_frames_task.pk, ], queue=settings.TASK_NAMES_TO_QUEUE[task_name])
    else:
        raise ValueError,"Extension {} not allowed".format(filename.split('.')[-1])
    return video