from __future__ import absolute_import
import subprocess, os, time, logging, requests, zipfile, io, sys, json, tempfile
from urlparse import urlparse
from collections import defaultdict
from PIL import Image
from django.conf import settings
from dva.celery import app
from .models import Video, Frame, TEvent,  IndexEntries, LOPQCodes, Region, Tube, \
    Retriever, Segment, QueryIndexVector, DeletedVideo, ManagementAction, SystemState, DVAPQL, \
    Worker, QueryRegion, QueryRegionIndexVector, DeepModel, RegionLabel, FrameLabel, Label

from .operations.indexing import IndexerTask
from .operations.retrieval import RetrieverTask
from .operations.detection import DetectorTask
from .operations.segmentation import SegmentorTask
from .operations.analysis import AnalyzerTask
from .operations.decoding import VideoDecoder
from .operations.dataset import DatasetCreator
from .processing import process_next, mark_as_completed
from dvalib import retriever
from django.utils import timezone
from celery.signals import task_prerun,celeryd_init
from . import serializers
from . import fs
from . import task_shared
try:
    import numpy as np
except ImportError:
    pass

W = None


@celeryd_init.connect
def configure_workers(sender, conf,**kwargs):
    global W
    W = Worker()
    W.pid = os.getpid()
    W.host = sender.split('.')[-1]
    W.queue_name = sender.split('@')[1].split('.')[0]
    W.save()


@task_prerun.connect
def start_task(task_id,task,args,**kwargs):
    if task.name.startswith('perform'):
        start = TEvent.objects.get(pk=args[0])
        start.task_id = task_id
        start.start_ts = timezone.now()
        if W and start.worker is None:
                start.worker_id = W.pk
        start.save()


@app.task(track_started=True, name="perform_map")
def perform_map(task_id):
    """
    map tasks on set of videos/datasets.
    :param task_id:
    :return:
    """
    raise NotImplementedError


@app.task(track_started=True, name="perform_image_download")
def perform_image_download(task_id):
    """
    Download images from remote path such as a url or an S3 bucket
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
    frame_list_json = args['frame_list_json']
    start_index = args['start_index']
    process_next(task_id)
    mark_as_completed(start)


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
    start.save()
    if 'index' in json_args:
        index_name = json_args['index']
        visual_index, di = perform_indexing.get_index_by_name(index_name)
    else:
        visual_index, di = perform_indexing.get_index_by_pk(json_args['indexer_pk'])
    sync = True
    if target == 'query':
        local_path = task_shared.download_and_get_query_path(start)
        vector = visual_index.apply(local_path)
        # TODO: figure out a better way to store numpy arrays.
        s = io.BytesIO()
        np.save(s,vector)
        # can be replaced by Redis instead of using DB
        _ = QueryIndexVector.objects.create(vector=s.getvalue(),event=start)
        sync = False
    elif target == 'query_regions':
        queryset, target = task_shared.build_queryset(args=start.arguments)
        region_paths = task_shared.download_and_get_query_region_path(start, queryset)
        for i,dr in enumerate(queryset):
            local_path = region_paths[i]
            vector = visual_index.apply(local_path)
            s = io.BytesIO()
            np.save(s,vector)
            # can be replaced by Redis instead of using DB
            _ = QueryRegionIndexVector.objects.create(vector=s.getvalue(),event=start,query_region=dr)
        sync = False
    elif target == 'regions':
        # For regions simply download/ensure files exists.
        queryset, target = task_shared.build_queryset(args=start.arguments, video_id=start.video_id)
        task_shared.ensure_files(queryset, target)
        perform_indexing.index_queryset(di,visual_index,start,target,queryset)
    elif target == 'frames':
        queryset, target = task_shared.build_queryset(args=start.arguments, video_id=start.video_id)
        if visual_index.cloud_fs_support and settings.DISABLE_NFS:
            # if NFS is disabled and index supports cloud file systems natively (e.g. like Tensorflow)
            perform_indexing.index_queryset(di, visual_index, start, target, queryset, cloud_paths=True)
        else:
            # Otherwise download and ensure that the files exist
            task_shared.ensure_files(queryset, target)
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
    target = args.get('target','query') # by default target is query
    if target == 'query':
        vector = np.load(io.BytesIO(QueryIndexVector.objects.get(event=start.parent_id).vector))
        perform_retrieval.retrieve(start,args.get('retriever_pk',20),vector,args.get('count',20))
    elif target == 'query_region_index_vectors':
        queryset, target = task_shared.build_queryset(args=args)
        for dr in queryset:
            vector = np.load(io.BytesIO(dr.vector))
            perform_retrieval.retrieve(start, args.get('retriever_pk', 20), vector, args.get('count', 20),
                                       region=dr.query_region)
    else:
        raise NotImplementedError,target
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
    task_shared.ensure('/{}/video/{}.zip'.format(video_id,video_id))
    dv.create_directory(create_subdirs=True)
    v = DatasetCreator(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.extract(start)
    process_next(task_id)
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
    task_shared.ensure(dv.path(media_root=''))
    dv.create_directory(create_subdirs=True)
    v = VideoDecoder(dvideo=dv, media_dir=settings.MEDIA_ROOT)
    v.get_metadata()
    v.segment_video(task_id)
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
    if 'target' not in args:
        args['target'] = 'segments'
    queryset, target = task_shared.build_queryset(args,video_id,start.parent_process_id)
    if target != 'segments':
        raise NotImplementedError("Cannot decode target:{}".format(target))
    task_shared.ensure_files(queryset,target)
    for ds in queryset:
        v.decode_segment(ds=ds,denominator=args.get('rate',30),event_id=task_id)
    process_next(task_id)
    mark_as_completed(start)
    return task_id


@app.task(track_started=True,name="perform_video_decode_lambda",ignore_result=False)
def perform_video_decode_lambda(task_id):
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
        request_args = dict(ds=ds,denominator=args['rate'],rescale=args['rescale'])
        # TODO : Implement this, call API gateway with video id, segment index and rate.
        #        If successful bulk create frame objects using response sent by the lambda function.
    process_next(task_id)
    mark_as_completed(start)
    raise NotImplementedError


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
    frame_detections_list = []
    dv = None
    dd_list = []
    query_flow = ('target' in args and args['target'] == 'query')
    if 'detector_pk' in args:
        detector_pk = int(args['detector_pk'])
        cd = DeepModel.objects.get(pk=detector_pk,model_type=DeepModel.DETECTOR)
        detector_name = cd.name
    else:
        detector_name = args['detector']
        cd = DeepModel.objects.get(name=detector_name,model_type=DeepModel.DETECTOR)
        detector_pk = cd.pk
    perform_detection.load_detector(cd)
    detector = perform_detection.get_static_detectors[cd.pk]
    if detector.session is None:
        logging.info("loading detection model")
        detector.load()
    if query_flow:
        local_path = task_shared.download_and_get_query_path(start)
        frame_detections_list.append((None,detector.detect(local_path)))
    else:
        if 'target' not in args:
            args['target'] = 'frames'
        dv = Video.objects.get(id=video_id)
        queryset, target = task_shared.build_queryset(args, video_id, start.parent_process_id)
        task_shared.ensure_files(queryset,target)
        for k in queryset:
            if target == 'frames':
                local_path = k.path()
            elif target == 'regions':
                local_path = k.frame_path()
            else:
                raise NotImplementedError("Invalid target:{}".format(target))
            frame_detections_list.append((k, detector.detect(local_path)))
    for df,detections in frame_detections_list:
        for d in detections:
            dd = QueryRegion() if query_flow else Region()
            dd.region_type = Region.DETECTION
            if query_flow:
                dd.query_id = start.parent_process_id
            else:
                dd.video_id = dv.pk
                dd.frame_id = df.pk
                dd.frame_index = df.frame_index
                dd.segment_index = df.segment_index
            if detector_name == 'coco':
                dd.object_name = '{}'.format(d['object_name'])
                dd.confidence = 100.0 * d['score']
            elif detector_name == 'textbox':
                dd.object_name = 'TEXTBOX'
                dd.confidence = 100.0 * d['score']
            elif detector_name == 'face':
                dd.object_name = 'MTCNN_face'
                dd.confidence = 100.0
            elif detector_name == 'custom':
                dd.object_name = '{}_{}'.format(detector_pk, d['object_name'])
                dd.confidence = 100.0 * d['score']
            else:
                raise NotImplementedError
            dd.x = d['x']
            dd.y = d['y']
            dd.w = d['w']
            dd.h = d['h']
            dd.event_id = task_id
            dd_list.append(dd)
    if query_flow:
        _ = QueryRegion.objects.bulk_create(dd_list, 1000)
    else:
        _ = Region.objects.bulk_create(dd_list, 1000)
    launched = process_next(task_id)
    mark_as_completed(start)
    if query_flow:
        return launched
    else:
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
        da = DeepModel.objects.get(name=analyzer_name,model_type=DeepModel.ANALYZER)
        perform_analysis.load_analyzer(da)
    analyzer = perform_analysis.get_static_analyzers[analyzer_name]
    regions_batch = []
    queryset, target = task_shared.build_queryset(args, video_id, start.parent_process_id)
    query_path = None
    query_regions_paths = None
    if target == 'query':
        query_path = task_shared.download_and_get_query_path(start)
    elif target == 'query_regions':
        query_regions_paths = task_shared.download_and_get_query_region_path(start, queryset)
    else:
        task_shared.ensure_files(queryset, target)
    image_data = {}
    frames_to_labels = []
    regions_to_labels = []
    labels_pk = {}
    temp_root = tempfile.mkdtemp()
    for i,f in enumerate(queryset):
        if query_regions_paths:
            path = query_regions_paths[i]
            a = QueryRegion()
            a.query_id = start.parent_process_id
            a.x = f.x
            a.y = f.y
            a.w = f.w
            a.h = f.h
        elif query_path:
            path = query_path
            w, h = task_shared.get_query_dimensions(start)
            a = QueryRegion()
            a.query_id = start.parent_process_id
            a.x = 0
            a.y = 0
            a.w = w
            a.h = h
            a.full_frame = True
        else:
            a = Region()
            a.video_id = f.video_id
            if target == 'regions':
                a.x = f.x
                a.y = f.y
                a.w = f.w
                a.h = f.h
                a.frame_id = f.frame.id
                a.frame_index = f.frame_index
                a.segment_index = f.segment_index
                path = task_shared.crop_and_get_region_path(f,image_data,temp_root)
            elif target == 'frames':
                a.full_frame = True
                a.frame_index = f.frame_index
                a.segment_index = f.segment_index
                a.frame_id = f.id
                path = f.path()
            else:
                raise NotImplementedError
        object_name, text, metadata, labels = analyzer.apply(path)
        for l in labels:
            if (l,analyzer.label_set) not in labels_pk:
                labels_pk[(l,analyzer.label_set)] = Label.objects.get_or_create(name=l,set=analyzer.label_set)[0].pk
            if target == 'regions':
                regions_to_labels.append(RegionLabel(label_id=labels_pk[(l,analyzer.label_set)],region_id=f.pk,
                                                     frame_id=f.frame.pk, frame_index=f.frame_index,
                                                     segment_index=f.segment_index,video_id=f.video_id,
                                                     event_id=task_id))
            elif target == 'frames':
                frames_to_labels.append(FrameLabel(label_id=labels_pk[(l, analyzer.label_set)],frame_id=f.pk,
                                                   frame_index=f.frame_index, segment_index=f.segment_index,
                                                   video_id=f.video_id, event_id=task_id))
        a.region_type = Region.ANNOTATION
        a.object_name = object_name
        a.text = text
        a.metadata = metadata
        a.event_id = task_id
        regions_batch.append(a)
    if query_regions_paths or query_path:
        QueryRegion.objects.bulk_create(regions_batch, 1000)
    else:
        Region.objects.bulk_create(regions_batch,1000)
    if regions_to_labels:
        RegionLabel.objects.bulk_create(regions_to_labels,1000)
    if frames_to_labels:
        FrameLabel.objects.bulk_create(frames_to_labels,1000)
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
    if settings.DISABLE_NFS:
        fs.download_video_from_remote_to_local(dv)
    destination = start.arguments['destination']
    try:
        if destination == "FILE":
                file_name = task_shared.export_file(dv, export_event_pk=start.pk)
                start.arguments['file_name'] = file_name
        elif destination == "S3":
            path = start.arguments['path']
            returncode = task_shared.perform_s3_export(dv, path, export_event_pk=start.pk)
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
    dd = DeepModel.objects.get(pk=start.arguments['detector_pk'],detector_type=DeepModel.DETECTOR)
    dd.create_directory(create_subdirs=False)
    if 'www.dropbox.com' in args['path'] and not args['path'].endswith('?dl=1'):
        r = requests.get(args['path'] + '?dl=1')
    else:
        r = requests.get(args['path'])
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


@app.task(track_started=True, name="perform_model_import")
def perform_model_import(task_id):
    start = TEvent.objects.get(pk=task_id)
    if start.started:
        return 0  # to handle celery bug with ACK in SOLO mode
    else:
        start.started = True
        start.save()
    args = start.arguments
    dm = DeepModel.objects.get(pk=args['pk'])
    dirname = 'models'
    task_shared.download_model(settings.MEDIA_ROOT,dirname, dm)
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
    path = start.arguments.get('path',None)
    dv = start.video
    if path.startswith('http'):
        u = urlparse(path)
        if u.hostname == 'www.youtube.com' or start.arguments.get('force_youtube_dl',False):
            task_shared.retrieve_video_via_url(start.video, settings.MEDIA_ROOT)
        else:
            task_shared.import_url(dv,start.arguments['path'])
    elif path.startswith('gs://') or path.startswith('s3://'):
        task_shared.import_remote(dv,path)
    elif path.startswith('/'):
        fs.ingest_path(dv,path)
    else:
        raise NotImplementedError('Unknown prefix {} must be one of  http, s3, gs or /'.format(path))
    if path.endswith('.json') or path.endswith('gz'):
        task_shared.load_frame_list(dv)
    elif path.endswith('.dva_export.zip') or path is None:
        task_shared.load_dva_export_file(dv)
    dv.uploaded = True
    dv.save()
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
    dc.last_built = timezone.now()
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
    if settings.MEDIA_BUCKET:
        dirname = args.get('dirname',None)
        task_shared.upload(dirname,start.parent_id,start.video_id)
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
def manage_host(self,op,ping_index=None,worker_name=None,queue_name=None):
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
    host_name = self.request.hostname
    if op == "list":
        ManagementAction.objects.create(op=op, parent_task=self.request.id, message="", host=host_name,
                                        ping_index=ping_index)
        for w in Worker.objects.filter(host=host_name.split('.')[-1], alive=True):
            # launch all queues EXCEPT worker processing manager queue
            if not task_shared.pid_exists(w.pid):
                w.alive = False
                w.save()
                for t in TEvent.objects.filter(started=True, completed=False, errored=False, worker=w):
                    t.errored = True
                    t.save()
                if w.queue_name != 'manager':
                    task_shared.launch_worker(w.queue_name, worker_name)
                    message = "worker processing {} is dead, restarting".format(w.queue_name)
                    ManagementAction.objects.create(op='worker_restart', parent_task=self.request.id,
                                                    message=message, host=host_name)
    elif op == "launch":
        if worker_name == host_name:
            message = task_shared.launch_worker(queue_name, worker_name)
            ManagementAction.objects.create(op='worker_launch', parent_task=self.request.id,
                                            message=message, host=host_name)
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


@app.task(track_started=True,name="monitor_system")
def monitor_system():
    """
    This task used by scheduler to monitor state of the system.
    :return:
    """
    for p in DVAPQL.objects.filter(completed=False):
        if TEvent.objects.filter(parent_process=p,completed=False).count() == 0:
            p.completed = True
            p.save()
    last_action = ManagementAction.objects.filter(ping_index__isnull=False).last()
    if last_action:
        ping_index = last_action.ping_index + 1
    else:
        ping_index = 0
    # TODO: Handle the case where host manager has not responded to last and itself has died
    _ = app.send_task('manage_host', args=['list', ping_index], exchange='qmanager')
    s = SystemState()
    s.processes = DVAPQL.objects.count()
    s.completed_processes = DVAPQL.objects.filter(completed=True).count()
    s.tasks = TEvent.objects.count()
    s.pending_tasks = TEvent.objects.filter(started=False).count()
    s.completed_tasks = TEvent.objects.filter(started=True,completed=True).count()
    s.save()