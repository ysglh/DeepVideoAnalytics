from django.conf import settings
from .operations import indexing, detection, analysis, approximation
import io, logging, tempfile, json, uuid
from . import task_shared
from . import models

try:
    import numpy as np
except ImportError:
    pass


def handle_perform_indexing(start):
    json_args = start.arguments
    target = json_args.get('target', 'frames')
    if 'index' in json_args:
        index_name = json_args['index']
        visual_index, di = indexing.Indexers.get_index_by_name(index_name)
    else:
        visual_index, di = indexing.Indexers.get_index_by_pk(json_args['indexer_pk'])
    sync = True
    if target == 'query':
        local_path = task_shared.download_and_get_query_path(start)
        vector = visual_index.apply(local_path)
        # TODO: figure out a better way to store numpy arrays.
        s = io.BytesIO()
        np.save(s, vector)
        # can be replaced by Redis instead of using DB
        _ = models.QueryIndexVector.objects.create(vector=s.getvalue(), event=start)
        sync = False
    elif target == 'query_regions':
        queryset, target = task_shared.build_queryset(args=start.arguments)
        region_paths = task_shared.download_and_get_query_region_path(start, queryset)
        for i, dr in enumerate(queryset):
            local_path = region_paths[i]
            vector = visual_index.apply(local_path)
            s = io.BytesIO()
            np.save(s, vector)
            # can be replaced by Redis instead of using DB
            _ = models.QueryRegionIndexVector.objects.create(vector=s.getvalue(), event=start, query_region=dr)
        sync = False
    elif target == 'regions':
        # For regions simply download/ensure files exists.
        queryset, target = task_shared.build_queryset(args=start.arguments, video_id=start.video_id)
        task_shared.ensure_files(queryset, target)
        indexing.Indexers.index_queryset(di, visual_index, start, target, queryset)
    elif target == 'frames':
        queryset, target = task_shared.build_queryset(args=start.arguments, video_id=start.video_id)
        if visual_index.cloud_fs_support and settings.DISABLE_NFS:
            # if NFS is disabled and index supports cloud file systems natively (e.g. like Tensorflow)
            indexing.Indexers.index_queryset(di, visual_index, start, target, queryset, cloud_paths=True)
        else:
            # Otherwise download and ensure that the files exist
            task_shared.ensure_files(queryset, target)
            indexing.Indexers.index_queryset(di, visual_index, start, target, queryset)
    return sync


def handle_perform_index_approximation(start):
    args = start.arguments
    if 'approximator_pk' in args:
        approx, da = approximation.Approximators.get_approximator_by_pk(args['approximator_pk'])
    elif 'approximator_shasum' in args:
        approx, da = approximation.Approximators.get_approximator_by_shasum(args['approximator_shasum'])
    else:
        raise ValueError("Could not find approximator {}".format(args))
    if args['target'] == 'index_entries':
        queryset, target = task_shared.build_queryset(args, start.video_id, start.parent_process_id)
        new_approx_indexes = []
        for index_entry in queryset:
            vectors, entries = index_entry.load_index()
            for i, e in enumerate(entries):
                e['codes'] = approx.approximate(vectors[i, :])
            approx_ind = models.IndexEntries()
            approx_ind.indexer_shasum = index_entry.indexer_shasum
            approx_ind.approximator_shasum = da.shasum
            approx_ind.count = index_entry.count
            approx_ind.approximate = True
            approx_ind.detection_name = index_entry.detection_name
            approx_ind.contains_detections = index_entry.contains_detections
            approx_ind.contains_frames = index_entry.contains_frames
            approx_ind.video_id = index_entry.video_id
            approx_ind.algorithm = da.name
            approx_ind.event_id = start.pk
            uid = str(uuid.uuid1()).replace('-', '_')
            entries_fname = "{}/{}/indexes/{}.json".format(settings.MEDIA_ROOT, start.video_id, uid)
            with open(entries_fname, 'w') as entryfile:
                json.dump(entries, entryfile)
            approx_ind.entries_file_name = "{}.json".format(uid)
            approx_ind.features_file_name = ""
            new_approx_indexes.append(approx_ind)
        models.IndexEntries.objects.bulk_create(new_approx_indexes, batch_size=100)
    else:
        raise ValueError("Target {} not allowed, only index_entries are allowed".format(args['target']))


def handle_perform_detection(start):
    video_id = start.video_id
    args = start.arguments
    frame_detections_list = []
    dv = None
    dd_list = []
    query_flow = ('target' in args and args['target'] == 'query')
    if 'detector_pk' in args:
        detector_pk = int(args['detector_pk'])
        cd = models.TrainedModel.objects.get(pk=detector_pk, model_type=models.TrainedModel.DETECTOR)
        detector_name = cd.name
    else:
        detector_name = args['detector']
        cd = models.TrainedModel.objects.get(name=detector_name, model_type=models.TrainedModel.DETECTOR)
        detector_pk = cd.pk
    detection.Detectors.load_detector(cd)
    detector = detection.Detectors._detectors[cd.pk]
    if detector.session is None:
        logging.info("loading detection model")
        detector.load()
    if query_flow:
        local_path = task_shared.download_and_get_query_path(start)
        frame_detections_list.append((None, detector.detect(local_path)))
    else:
        if 'target' not in args:
            args['target'] = 'frames'
        dv = models.Video.objects.get(id=video_id)
        queryset, target = task_shared.build_queryset(args, video_id, start.parent_process_id)
        task_shared.ensure_files(queryset, target)
        for k in queryset:
            if target == 'frames':
                local_path = k.path()
            elif target == 'regions':
                local_path = k.frame_path()
            else:
                raise NotImplementedError("Invalid target:{}".format(target))
            frame_detections_list.append((k, detector.detect(local_path)))
    for df, detections in frame_detections_list:
        for d in detections:
            dd = models.QueryRegion() if query_flow else models.Region()
            dd.region_type = models.Region.DETECTION
            if query_flow:
                dd.query_id = start.parent_process_id
            else:
                dd.video_id = dv.pk
                dd.frame_id = df.pk
                dd.frame_index = df.frame_index
                dd.segment_index = df.segment_index
            if detector_name == 'textbox':
                dd.object_name = 'TEXTBOX'
                dd.confidence = 100.0 * d['score']
            elif detector_name == 'face':
                dd.object_name = 'MTCNN_face'
                dd.confidence = 100.0
            else:
                dd.object_name = d['object_name']
                dd.confidence = 100.0 * d['score']
            dd.x = d['x']
            dd.y = d['y']
            dd.w = d['w']
            dd.h = d['h']
            dd.event_id = start.pk
            dd_list.append(dd)
    if query_flow:
        _ = models.QueryRegion.objects.bulk_create(dd_list, 1000)
    else:
        _ = models.Region.objects.bulk_create(dd_list, 1000)
    return query_flow


def handle_perform_analysis(start):
    task_id = start.pk
    video_id = start.video_id
    args = start.arguments
    analyzer_name = args['analyzer']
    if analyzer_name not in analysis.Analyzers._analyzers:
        da = models.TrainedModel.objects.get(name=analyzer_name, model_type=models.TrainedModel.ANALYZER)
        analysis.Analyzers.load_analyzer(da)
    analyzer = analysis.Analyzers._analyzers[analyzer_name]
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
    for i, f in enumerate(queryset):
        if query_regions_paths:
            path = query_regions_paths[i]
            a = models.QueryRegion()
            a.query_id = start.parent_process_id
            a.x = f.x
            a.y = f.y
            a.w = f.w
            a.h = f.h
        elif query_path:
            path = query_path
            w, h = task_shared.get_query_dimensions(start)
            a = models.QueryRegion()
            a.query_id = start.parent_process_id
            a.x = 0
            a.y = 0
            a.w = w
            a.h = h
            a.full_frame = True
        else:
            a = models.Region()
            a.video_id = f.video_id
            if target == 'regions':
                a.x = f.x
                a.y = f.y
                a.w = f.w
                a.h = f.h
                a.frame_id = f.frame.id
                a.frame_index = f.frame_index
                a.segment_index = f.segment_index
                path = task_shared.crop_and_get_region_path(f, image_data, temp_root)
            elif target == 'frames':
                a.full_frame = True
                a.frame_index = f.frame_index
                a.segment_index = f.segment_index
                a.frame_id = f.id
                path = f.path()
            else:
                raise NotImplementedError
        object_name, text, metadata, labels = analyzer.apply(path)
        if labels:
            for l in labels:
                if (l, analyzer.label_set) not in labels_pk:
                    labels_pk[(l, analyzer.label_set)] = models.Label.objects.get_or_create(name=l,
                                                                                             set=analyzer.label_set)[0].pk
                if target == 'regions':
                    regions_to_labels.append(
                        models.RegionLabel(label_id=labels_pk[(l, analyzer.label_set)], region_id=f.pk,
                                           frame_id=f.frame.pk, frame_index=f.frame_index,
                                           segment_index=f.segment_index, video_id=f.video_id,
                                           event_id=task_id))
                elif target == 'frames':
                    frames_to_labels.append(
                        models.FrameLabel(label_id=labels_pk[(l, analyzer.label_set)], frame_id=f.pk,
                                          frame_index=f.frame_index, segment_index=f.segment_index,
                                          video_id=f.video_id, event_id=task_id))
        a.region_type = models.Region.ANNOTATION
        a.object_name = object_name
        a.text = text
        a.metadata = metadata
        a.event_id = task_id
        regions_batch.append(a)
    if query_regions_paths or query_path:
        models.QueryRegion.objects.bulk_create(regions_batch, 1000)
    else:
        models.Region.objects.bulk_create(regions_batch, 1000)
    if regions_to_labels:
        models.RegionLabel.objects.bulk_create(regions_to_labels, 1000)
    if frames_to_labels:
        models.FrameLabel.objects.bulk_create(frames_to_labels, 1000)
