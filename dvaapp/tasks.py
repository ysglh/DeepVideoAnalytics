from __future__ import absolute_import
import subprocess,sys,shutil,os,glob,time,logging
from django.conf import settings
from celery import shared_task
from .models import Video, Frame, Detection, TEvent, Query, IndexEntries,QueryResults, FrameLabel
from dvalib import entity
from dvalib import detector
from PIL import Image
import json
import zipfile



@shared_task
def perform_indexing(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = "indexing"
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    video = entity.WVideo(dv, settings.MEDIA_ROOT)
    frames = Frame.objects.all().filter(video=dv)
    for index_results in video.index_frames(frames):
        i = IndexEntries()
        i.video = dv
        i.count = index_results['count']
        i.algorithm = index_results['index_name']
        i.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@shared_task
def query_by_image(query_id):
    dq = Query.objects.get(id=query_id)
    start = TEvent()
    start.video_id = Video.objects.get(parent_query=dq).pk
    start.started = True
    start.operation = "query"
    start.save()
    start_time = time.time()
    Q = entity.WQuery(dquery=dq, media_dir=settings.MEDIA_ROOT)
    results = Q.find()
    dq.results = True
    dq.results_metadata = json.dumps(results)
    for algo,rlist in results.iteritems():
        for r in rlist:
            qr = QueryResults()
            qr.query = dq
            qr.frame_id = r['frame_primary_key']
            qr.video_id = r['video_primary_key']
            qr.algorithm = algo
            qr.rank = r['rank']
            qr.distance = r['dist']
            qr.save()
    dq.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return results


@shared_task
def extract_frames(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = "extract_frames"
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    v = entity.WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    time.sleep(3) # otherwise ffprobe randomly fails
    if not dv.dataset:
        v.get_metadata()
        dv.metadata = v.metadata
        dv.length_in_seconds = v.duration
        dv.height = v.height
        dv.width = v.width
        dv.save()
    frames = v.extract_frames()
    dv.frames = len(frames)
    dv.save()
    for f in frames:
        df = Frame()
        df.frame_index = f.frame_index
        df.video = dv
        if f.name:
            df.name = f.name[:150]
            df.subdir = f.subdir.replace('/',' ')
        df.save()
        if f.name:
            for l in f.subdir.split('/'):
                if l != dv.name and l.strip():
                    fl = FrameLabel()
                    fl.frame = df
                    fl.label = l
                    fl.video = dv
                    fl.source = "directory_name"
                    fl.save()
    perform_indexing.apply_async(args=[video_id],queue=settings.Q_INDEXER)
    perform_detection.apply_async(args=[video_id], queue=settings.Q_DETECTOR)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@shared_task
def perform_detection(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = "detection"
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    frames = Frame.objects.all().filter(video=dv)
    v = entity.WVideo(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    wframes = {df.pk: entity.WFrame(video=v, frame_index=df.frame_index, primary_key=df.pk) for df in frames}
    detection_count = 0
    for alogrithm in detector.DETECTORS.itervalues():
        frame_detections = alogrithm.detect(wframes.values())
        print frame_detections.items()
        for frame_pk,detections in frame_detections.iteritems():
            for d in detections:
                dd = Detection()
                dd.video = dv
                dd.frame_id = frame_pk
                dd.object_name = d['name']
                dd.confidence = d['confidence']
                dd.x = d['left']
                dd.y = d['top']
                dd.w = d['right'] - d['left']
                dd.h = d['bot'] - d['top']
                dd.save()
                img = Image.open(wframes[frame_pk].local_path())
                img2 = img.crop((d['left'], d['top'], d['right'], d['bot']))
                img2.save("{}/{}/detections/{}.jpg".format(settings.MEDIA_ROOT, video_id, dd.pk))
                detection_count += 1
    dv.detections = detection_count
    dv.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0

