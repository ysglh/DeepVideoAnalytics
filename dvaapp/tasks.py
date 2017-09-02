from __future__ import absolute_import
import subprocess, os, time, logging, requests, zipfile, io, sys, json
from collections import defaultdict
from PIL import Image
from django.conf import settings
from dva.celery import app
from .models import Video, Frame, TEvent,  IndexEntries, LOPQCodes, Region, Tube, \
    Retriever, Detector, Segment, QueryIndexVector, DeletedVideo, ManagementAction, Indexer, Analyzer
from .operations.indexing import IndexerTask
from .operations.retrieval import RetrieverTask
from .operations.detection import DetectorTask
from .operations.segmentation import SegmentorTask
from .operations.analysis import AnalyzerTask
from .operations.decoding import VideoDecoder
from .operations.processing import process_next, mark_as_completed
from dvalib import retriever
from django.utils import timezone
from celery.signals import task_prerun
from . import shared
try:
    import numpy as np
except ImportError:
    pass
from . import serializers


@task_prerun.connect
def start_task(task_id,task,args,**kwargs):
    if task.name.startswith('perform'):
        start = TEvent.objects.get(pk=args[0])
        start.task_id = task_id
        start.start_ts = timezone.now()
        start.save()


@app.task(track_started=True, name="perform_indexing", base=IndexerTask)
def perform_indexing(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    json_args = start.arguments
    target = json_args.get('target','frames')
    index_name = json_args['index']
    start.save()
    visual_index, di = perform_indexing.get_index_by_name(index_name)
    sync = True
    if target == 'query':
        parent_process = start.parent_process
        local_path = "{}/queries/{}_{}.png".format(settings.MEDIA_ROOT, start.pk, start.parent_process_id)
        with open(local_path, 'w') as fh:
            fh.write(str(parent_process.image_data))
        vector = visual_index.apply(local_path)
        # TODO: figure out a better way to store numpy arrays.
        s = io.BytesIO()
        np.save(s,vector)
        _ = QueryIndexVector.objects.create(vector=s.getvalue(),event=start)
        sync = False
    else:
        queryset, target = shared.build_queryset(args=start.arguments,video_id=start.video_id)
        perform_indexing.index_queryset(di,visual_index,start,target,queryset)
    next_ids = process_next(task_id,sync=sync)
    mark_as_completed(start)
    return next_ids


@app.task(track_started=True, name="perform_transformation")
def perform_transformation(task_id):
    """
    Crop detected or annotated regions
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    resize = args.get('resize',None)
    kwargs = args.get('filters',{})
    paths_to_regions = defaultdict(list)
    kwargs['video_id'] = start.video_id
    kwargs['materialized'] = False
    logging.info("executing crop with kwargs {}".format(kwargs))
    queryset = Region.objects.all().filter(**kwargs)
    for dr in queryset:
        paths_to_regions[dr.frame_path()].append(dr)
    for path,regions in paths_to_regions.iteritems():
        img = Image.open(path)
        for dr in regions:
            cropped = img.crop((dr.x, dr.y,dr.x + dr.w, dr.y + dr.h))
            if resize:
                resized = cropped.resize(tuple(resize),Image.BICUBIC)
                resized.save(dr.path())
            else:
                cropped.save(dr.path())
    queryset.update(materialized=True)
    process_next(task_id)
    mark_as_completed(start)


@app.task(track_started=True, name="perform_retrieval", base=RetrieverTask)
def perform_retrieval(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    vector = np.load(io.BytesIO(QueryIndexVector.objects.get(event=start.parent_id).vector))
    perform_retrieval.retrieve(start,args.get('retriever_pk',20),vector,args.get('count',20))
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_dataset_extraction")
def perform_dataset_extraction(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    if args == {}:
        args['rescale'] = 0
        args['rate'] = 30
        start.arguments = args
    start.save()
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    if dv.youtube_video:
        dv.create_directory()
    v = VideoDecoder(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.extract(args=args,start=start)
    process_next(task_id)
    os.remove("{}/{}/video/{}.zip".format(settings.MEDIA_ROOT, dv.pk, dv.pk))
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_video_segmentation")
def perform_video_segmentation(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    if 'rescale' not in args:
        args['rescale'] = 0
    if 'rate' not in args:
        args['rate'] = 30
    start.arguments = args
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    v = VideoDecoder(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.get_metadata()
    v.segment_video()
    if args.get('sync',False):
        next_args = {'rescale': args['rescale'], 'rate': args['rate']}
        next_task = TEvent.objects.create(video=dv, operation='perform_video_decode', arguments=next_args, parent=start)
        perform_video_decode(next_task.pk)  # decode it synchronously for testing in Travis
        process_next(task_id,sync=True,launch_next=False)
    else:
        process_next(task_id)
    mark_as_completed(start)
    return 0


@app.task(track_started=True,name="perform_video_decode",ignore_result=False)
def perform_video_decode(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    video_id = start.video_id
    dv = Video.objects.get(id=video_id)
    kwargs = args.get('filters',{})
    kwargs['video_id'] = video_id
    v = VideoDecoder(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    for ds in Segment.objects.filter(**kwargs):
        v.decode_segment(ds=ds,denominator=args['rate'],rescale=args['rescale'])
    process_next(task_id)
    mark_as_completed(start)
    return task_id


@app.task(track_started=True, name="perform_detection",base=DetectorTask)
def perform_detection(task_id):
    """
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    video_id = start.video_id
    args = start.arguments
    detector_name = args['detector']
    if 'detector_pk' in args:
        detector_pk = int(args['detector_pk'])
        cd = Detector.objects.get(pk=detector_pk)
    else:
        cd = Detector.objects.get(name=detector_name)
    perform_detection.load_detector(cd)
    detector = perform_detection.get_static_detectors[cd.pk]
    if detector.session is None:
        logging.info("loading detection model")
        detector.load()
    dv = Video.objects.get(id=video_id)
    if 'filters' in args:
        kwargs = args['filters']
        kwargs['video_id'] = video_id
        frames = Frame.objects.all().filter(**kwargs)
        logging.info("Running {} Using filters {}".format(detector_name,kwargs))
    else:
        frames = Frame.objects.all().filter(video=dv)
    dd_list = []
    path_list = []
    for df in frames:
        local_path = df.path()
        detections = detector.detect(local_path)
        for d in detections:
            dd = Region()
            dd.region_type = Region.DETECTION
            dd.video_id = dv.pk
            dd.frame_id = df.pk
            dd.parent_frame_index = df.frame_index
            dd.parent_segment_index = df.segment_index
            if detector_name == 'coco':
                dd.object_name = 'SSD_{}'.format(d['object_name'])
                dd.confidence = 100.0 * d['score']
            elif detector_name == 'textbox':
                dd.object_name = 'TEXTBOX'
                dd.confidence = 100.0 * d['score']
            elif detector_name == 'face':
                dd.object_name = 'MTCNN_face'
                dd.confidence = 100.0
            elif detector_name == 'custom':
                dd.object_name = '{}_{}'.format(detector_pk,d['object_name'])
                dd.confidence = 100.0 * d['score']
            else:
                raise NotImplementedError
            dd.x = d['x']
            dd.y = d['y']
            dd.w = d['w']
            dd.h = d['h']
            dd.event_id = task_id
            dd_list.append(dd)
            path_list.append(local_path)
    _ = Region.objects.bulk_create(dd_list,1000)
    process_next(task_id)
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_analysis",base=AnalyzerTask)
def perform_analysis(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    video_id = start.video_id
    args = start.arguments
    analyzer_name = args['analyzer']
    if analyzer_name not in perform_analysis._analyzers:
        da = Analyzer.objects.get(name=analyzer_name)
        perform_analysis.load_analyzer(da)
    analyzer = perform_analysis.get_static_analyzers[analyzer_name]
    regions_batch = []
    queryset, target = shared.build_queryset(args,video_id)
    for f in queryset:
        path = f.path()
        object_name, text, metadata = analyzer.apply(path)
        a = Region()
        a.region_type = Region.ANNOTATION
        a.object_name = object_name
        a.text = text
        a.metadata = metadata
        a.event_id = task_id
        a.video_id = f.video_id
        if target == 'regions':
            a.x = f.x
            a.y = f.y
            a.w = f.w
            a.h = f.h
            a.frame_id = f.frame.id
            a.parent_frame_index = f.parent_frame_index
            a.parent_segment_index = f.parent_segment_index
        elif target == 'frames':
            a.full_frame = True
            a.parent_frame_index = f.frame_index
            a.parent_segment_index = f.segment_index
            a.frame_id = f.id
        regions_batch.append(a)
    Region.objects.bulk_create(regions_batch,1000)
    process_next(task_id)
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_export")
def perform_export(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    video_id = start.video_id
    dv = Video.objects.get(pk=video_id)
    destination = start.arguments['destination']
    try:
        if destination == "FILE":
                file_name = shared.export_file(dv,export_event_pk=start.pk)
                start.arguments['file_name'] = file_name
        elif destination == "S3":
            s3bucket = start.arguments['bucket']
            s3region = start.arguments['region']
            create_bucket = start.arguments.get('create_bucket',False)
            s3key = start.arguments['key']
            returncode = shared.perform_s3_export(dv,s3key,s3bucket,s3region, export_event_pk=start.pk,create_bucket=create_bucket)
            if returncode != 0:
                raise ValueError,"return code != 0"
    except:
        start.errored = True
        start.error_message = "Could not export"
        start.duration = (timezone.now() - start.start_ts).total_seconds()
        start.save()
        exc_info = sys.exc_info()
        raise exc_info[0], exc_info[1], exc_info[2]
    mark_as_completed(start)

@app.task(track_started=True, name="perform_detector_import")
def perform_detector_import(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    dd = Detector.objects.get(pk=start.arguments['detector_pk'])
    dd.create_directory(create_subdirs=False)
    if 'www.dropbox.com' in args['download_url'] and not args['download_url'].endswith('?dl=1'):
        r = requests.get(args['download_url'] + '?dl=1')
    else:
        r = requests.get(args['download_url'])
    output_filename = "{}/detectors/{}.zip".format(settings.MEDIA_ROOT, dd.pk)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()
    source_zip = "{}/detectors/{}.zip".format(settings.MEDIA_ROOT, dd.pk)
    zipf = zipfile.ZipFile(source_zip, 'r')
    zipf.extractall("{}/detectors/{}/".format(settings.MEDIA_ROOT, dd.pk))
    zipf.close()
    serializers.import_detector(dd)
    dd.save()
    os.remove(source_zip)
    process_next(task_id)
    mark_as_completed(start)

@app.task(track_started=True, name="perform_import")
def perform_import(event_id):
    start = TEvent.objects.get(pk=event_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    source = start.arguments['source']
    dv = start.video
    if source == 'URL':
        if start.video is None:
            start.video = shared.handle_video_url(start.arguments['name'],start.arguments['url'])
            start.save()
        shared.retrieve_video_via_url(start.video,settings.MEDIA_ROOT)
    elif source == 'S3':
        shared.import_s3(start,dv)
    elif source == 'VDN_URL':
        shared.import_vdn_url(dv,start.arguments['url'])
    elif source == 'VDN_S3':
        shared.import_vdn_s3(dv,start.arguments['key'],start.arguments['bucket'])
    elif source == 'LOCAL':
        shared.import_local(dv)
    elif source == 'MASSIVE':
        shared.import_external(start.arguments)
    else:
        raise NotImplementedError
    process_next(start.pk)
    mark_as_completed(start)


@app.task(track_started=True, name="perform_retriever_creation")
def perform_retriever_creation(cluster_task_id, test=False):
    start = TEvent.objects.get(pk=cluster_task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    dc = Retriever.objects.get(pk=start.arguments['retriever_pk'])
    dc.create_directory()
    dc.arguments['fnames'] = [ i.npy_path() for i in IndexEntries.objects.filter(**dc.source_filters) ]
    dc.arguments['proto_filename'] = dc.proto_filename()
    c = retriever.LOPQRetriever(name=dc.name,args=dc.arguments)
    c.cluster()
    cluster_codes = []
    for e in c.entries:
        cc = LOPQCodes()
        cc.retriever_id = dc.pk
        cc.video_id = e['video_primary_key']
        if 'detection_primary_key' in e:
            cc.detection_id = e['detection_primary_key']
            cc.frame_id = Region.objects.get(pk=cc.detection_id).frame_id
        else:
            cc.frame_id = e['frame_primary_key']
        cc.clusters = dc
        cc.coarse = e['coarse']
        cc.fine = e['fine']
        cc.coarse_text = " ".join(map(str, e['coarse']))
        cc.fine_text = " ".join(map(str, e['fine']))
        cc.searcher_index = e['index']
        cluster_codes.append(cc)
    LOPQCodes.objects.bulk_create(cluster_codes)
    c.save()
    dc.completed = True
    dc.save()
    mark_as_completed(start)


@app.task(track_started=True, name="perform_sync")
def perform_sync(task_id):
    """
    TODO: Determine a way to rate limit consecutive sync bucket for a given
    (video,directory). As an alternative perform sync at a more granular level,
    e.g. individual files. This is most important when syncing regions and frames.
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    video_id = start.video_id
    args = start.arguments
    if settings.MEDIA_BUCKET.strip():
        if 'dirname' in args:
            src = '{}/{}/{}/'.format(settings.MEDIA_ROOT, video_id, args['dirname'])
            dest = 's3://{}/{}/{}/'.format(settings.MEDIA_BUCKET, video_id, args['dirname'])
        else:
            src = '{}/{}/'.format(settings.MEDIA_ROOT, video_id)
            dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, video_id)
        command = " ".join(['aws', 's3', 'sync','--quiet', src, dest])
        syncer = subprocess.Popen(['aws', 's3', 'sync','--quiet', '--size-only', src, dest])
        syncer.wait()
        if syncer.returncode != 0:
            start.errored = True
            start.error_message = "Error while executing : {}".format(command)
            start.save()
            return
    else:
        logging.info("Media bucket name not specified, nothing was synced.")
        start.error_message = "Media bucket name is empty".format(settings.MEDIA_BUCKET)
    mark_as_completed(start)
    return


@app.task(track_started=True, name="perform_deletion")
def perform_deletion(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    video_pk = int(args['video_pk'])
    deleter_pk = args.get('deleter_pk',None)
    video = Video.objects.get(pk=video_pk)
    deleted = DeletedVideo()
    deleted.name = video.name
    deleted.deleter_id = deleter_pk
    deleted.uploader = video.uploader
    deleted.url = video.url
    deleted.description = video.description
    deleted.original_pk = video_pk
    deleted.save()
    video.delete()
    src = '{}/{}/'.format(settings.MEDIA_ROOT, int(video_pk))
    args = ['rm','-rf',src]
    command = " ".join(args)
    deleter = subprocess.Popen(args)
    deleter.wait()
    if deleter.returncode != 0:
        start.errored = True
        start.error_message = "Error while executing : {}".format(command)
        start.save()
        return
    if settings.MEDIA_BUCKET:
        dest = 's3://{}/{}/'.format(settings.MEDIA_BUCKET, int(video_pk))
        args = ['aws', 's3', 'rm','--quiet','--recursive', dest]
        command = " ".join(args)
        syncer = subprocess.Popen(args)
        syncer.wait()
        if syncer.returncode != 0:
            start.errored = True
            start.error_message = "Error while executing : {}".format(command)
            start.save()
            return
    else:
        logging.info("Media bucket name not specified, nothing was synced.")
        start.error_message = "Media bucket name is empty".format(settings.MEDIA_BUCKET)
    mark_as_completed(start)
    return


@app.task(track_started=True, name="perform_detector_training")
def perform_detector_training(task_id):
    """
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    train_detector = subprocess.Popen(['fab', 'train_yolo:{}'.format(start.pk)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0], '../'))
    train_detector.wait()
    if train_detector.returncode != 0:
        start.errored = True
        start.error_message = "fab train_yolo:{} failed with return code {}".format(start.pk,train_detector.returncode)
        start.duration = (timezone.now() - start.start_ts).total_seconds()
        start.save()
        raise ValueError, start.error_message
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_segmentation",base=SegmentorTask)
def perform_segmentation(task_id):
    """
    :param task_id:
    :return:
    """
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    video_id = start.video_id
    args = start.arguments
    segmentor_name = args['segmentor']
    segmentor = perform_segmentation.get_static_segmentors[segmentor_name]
    if segmentor.session is None:
        logging.info("loading detection model")
        segmentor.load()
    dv = Video.objects.get(id=video_id)
    if 'filters' in args:
        kwargs = args['filters']
        kwargs['video_id'] = video_id
        frames = Frame.objects.all().filter(**kwargs)
        logging.info("Running {} Using filters {}".format(segmentor_name,kwargs))
    else:
        frames = Frame.objects.all().filter(video=dv)
    dd_list = []
    path_list = []
    for df in frames:
        local_path = df.path()
        segmentation = segmentor.detect(local_path)
        # for d in detections:
        #     dd = Region()
        #     dd.region_type = Region.DETECTION
        #     dd.video_id = dv.pk
        #     dd.frame_id = df.pk
        #     dd.parent_frame_index = df.frame_index
        #     dd.parent_segment_index = df.segment_index
        #     if detector_name == 'coco':
        #         dd.object_name = 'SSD_{}'.format(d['object_name'])
        #         dd.confidence = 100.0 * d['score']
        #     elif detector_name == 'textbox':
        #         dd.object_name = 'TEXTBOX'
        #         dd.confidence = 100.0 * d['score']
        #     elif detector_name == 'face':
        #         dd.object_name = 'MTCNN_face'
        #         dd.confidence = 100.0
        #     else:
        #         raise NotImplementedError
        #     dd.x = d['x']
        #     dd.y = d['y']
        #     dd.w = d['w']
        #     dd.h = d['h']
        #     dd.event_id = task_id
        #     dd_list.append(dd)
        #     path_list.append(local_path)
    _ = Region.objects.bulk_create(dd_list,1000)
    process_next(task_id)
    mark_as_completed(start)
    return 0


@app.task(track_started=True, name="perform_compression")
def perform_compression(task_id):
    """
    TODO Compress a video or a dataset by removing all materialized regions
    and frames/segments (for videos). While retaining metadata and indexes.
    :param task_id:
    :return:
    """
    raise NotImplementedError


@app.task(track_started=True, name="perform_decompression")
def perform_decompression(task_id):
    """
    TODO De-compress a compressed video or a dataset by re-creating all materialized regions
    and frames/segments (for videos). Implementing this tasks correctly requires, exact
    FFmpeg version otherwise the segements might be split at different frames.
    :param task_id:
    :return:
    """
    raise NotImplementedError


@app.task(track_started=True,name="manage_host",bind=True)
def manage_host(self,op,worker_name=None,queue_name=None):
    """
    Manage host
    This task is handled by workers consuming from a broadcast management queue.
    It  allows quick inspection of GPU memory utilization launch of additional queues.
    Since TensorFlow workers need to be in SOLO concurrency mode, having additional set of workers
    enables easy management without a long timeout.
    Example use
    1. Launch worker to consume from a specific queue
    2. Gather GPU memory utilization info
    """
    message = ""
    host_name = self.request.hostname
    if op == "list":
        message = "test"
    elif op == "launch":
        if worker_name == host_name:
            p = subprocess.Popen(['fab','startq:{}'.format(queue_name)],close_fds=True)
            message = "launched {} with pid {} on {}".format(queue_name,p.pid,worker_name)
        else:
            message = "{} on {} ignored".format(queue_name, worker_name)
    elif op == "gpuinfo":
        try:
            message = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv']).splitlines()[1]
        except:
            message = "No GPU available"
    ManagementAction.objects.create(op=op,parent_task=self.request.id,message=message,host=host_name)


@app.task(track_started=True,name="perform_ingestion")
def perform_ingestion(task_id):
    """
    Incrementally add segments/images to video/dataset.
    Used for ingesting video from streaming sources such as
    cameras, online livestreams,twitter feed of a developing event, etc.
    :param task_id:
    :return:
    """
    raise NotImplementedError


@app.task(track_started=True,name="perform_stream_capture")
def perform_stream_capture(task_id):
    """
    Capture camera or livestream
    'livestreamer --player-continuous-http --player-no-close "{}" best -O --yes-run-as-root'.format(url)
    'ffmpeg -re -i - -c:v libx264 -c:a aac -ac 1 -strict -2 -crf 18 -profile:v baseline -maxrate 3000k
    -bufsize 1835k -pix_fmt yuv420p -flags -global_header -f segment -segment_time 0.1 "{}/%d.mp4"'.format(dirname)
    :param task_id:
    :return:
    """
    raise NotImplementedError