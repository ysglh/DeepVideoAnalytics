from __future__ import absolute_import
import subprocess,sys,shutil,os,glob,time,logging
from django.conf import settings
from dva.celery import app
from .models import Video, Frame, Detection, TEvent, Query, IndexEntries,QueryResults, Annotation, VLabel, Export
from dvalib import entity
from dvalib import detector
from dvalib import indexer
from collections import defaultdict
import calendar
from PIL import Image
from scipy import misc
import json
import celery
import zipfile
from .serializers import VideoExportSerializer

def process_video_next(video_id,current_task_name):
    if current_task_name in settings.POST_OPERATION_TASKS:
        for k in settings.POST_OPERATION_TASKS[current_task_name]:
            app.send_task(k,args=[video_id,],queue=settings.TASK_NAMES_TO_QUEUE[k])


class IndexerTask(celery.Task):
    _visual_indexer = None

    @property
    def visual_indexer(self):
        if IndexerTask._visual_indexer is None:
            IndexerTask._visual_indexer = {'inception': indexer.InceptionIndexer(),
                                           'facenet': indexer.FacenetIndexer(),
                                           'alexnet': indexer.AlexnetIndexer()}
        return IndexerTask._visual_indexer

    def refresh_index(self,index_name):
        index_entries = IndexEntries.objects.all()
        visual_index = self.visual_indexer[index_name]
        for index_entry in index_entries:
            if index_entry.pk not in visual_index.loaded_entries and index_entry.algorithm == index_name:
                fname = "{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id, index_entry.features_file_name)
                vectors = indexer.np.load(fname)
                vector_entries = json.load(file("{}/{}/indexes/{}".format(settings.MEDIA_ROOT, index_entry.video_id, index_entry.entries_file_name)))
                logging.info("Starting {} in {}".format(index_entry.video_id, visual_index.name))
                try:
                    visual_index.load_index(vectors, vector_entries)
                except:
                    logging.info("ERROR Failed to load {} ".format(index_entry.video_id))
                visual_index.loaded_entries.add(index_entry.pk)
                logging.info("finished {} in {}, current shape {}".format(index_entry.video_id, visual_index.name,visual_index.index.shape))


@app.task(name="inception_index_by_id",base=IndexerTask)
def inception_index_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = inception_index_by_id.name
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    video = entity.WVideo(dv, settings.MEDIA_ROOT)
    frames = Frame.objects.all().filter(video=dv)
    visual_index = inception_index_by_id.visual_indexer['inception']
    index_name, index_results, feat_fname, entries_fname = video.index_frames(frames,visual_index)
    i = IndexEntries()
    i.video = dv
    i.count = len(index_results)
    i.contains_frames = True
    i.detection_name = 'Frame'
    i.algorithm = index_name
    i.entries_file_name = entries_fname.split('/')[-1]
    i.features_file_name = feat_fname.split('/')[-1]
    i.save()
    process_video_next(video_id, start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(name="inception_index_ssd_detection_by_id",base=IndexerTask)
def inception_index_ssd_detection_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = inception_index_ssd_detection_by_id.name
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    video = entity.WVideo(dv, settings.MEDIA_ROOT)
    detections = Detection.objects.all().filter(video=dv,object_name__startswith='SSD_',w__gte=50,h__gte=50)
    logging.info("Indexing {} SSD detections".format(detections.count()))
    visual_index = inception_index_ssd_detection_by_id.visual_indexer['inception']
    index_name, index_results, feat_fname, entries_fname = video.index_detections(detections,'SSD',visual_index)
    i = IndexEntries()
    i.video = dv
    i.count = len(index_results)
    i.contains_detections = True
    i.detection_name = 'SSD'
    i.algorithm = index_name
    i.entries_file_name = entries_fname.split('/')[-1]
    i.features_file_name = feat_fname.split('/')[-1]
    i.save()
    process_video_next(video_id, start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(name="alexnet_index_by_id",base=IndexerTask)
def alexnet_index_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = alexnet_index_by_id.name
    start.save()
    start_time = time.time()
    dv = Video.objects.get(id=video_id)
    video = entity.WVideo(dv, settings.MEDIA_ROOT)
    frames = Frame.objects.all().filter(video=dv)
    visual_index = alexnet_index_by_id.visual_indexer['alexnet']
    index_name, index_results, feat_fname, entries_fname = video.index_frames(frames,visual_index)
    i = IndexEntries()
    i.video = dv
    i.count = len(index_results)
    i.contains_frames = True
    i.detection_name = 'Frame'
    i.algorithm = index_name
    i.entries_file_name = entries_fname.split('/')[-1]
    i.features_file_name = feat_fname.split('/')[-1]
    i.save()
    process_video_next(video_id, start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()


@app.task(name="inception_query_by_image",base=IndexerTask)
def inception_query_by_image(query_id):
    dq = Query.objects.get(id=query_id)
    start = TEvent()
    start.video_id = Video.objects.get(parent_query=dq).pk
    start.started = True
    start.operation = inception_query_by_image.name
    start.save()
    start_time = time.time()
    inception_query_by_image.refresh_index('inception')
    inception = inception_query_by_image.visual_indexer['inception']
    Q = entity.WQuery(dquery=dq, media_dir=settings.MEDIA_ROOT,visual_index=inception)
    results = Q.find(10)
    dq.results = True
    dq.results_metadata = json.dumps(results)
    for algo,rlist in results.iteritems():
        for r in rlist:
            qr = QueryResults()
            qr.query = dq
            if 'detection_primary_key' in r:
                qr.detection_id = r['detection_primary_key']
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


@app.task(name="alexnet_query_by_image",base=IndexerTask)
def alexnet_query_by_image(query_id):
    dq = Query.objects.get(id=query_id)
    start = TEvent()
    start.video_id = Video.objects.get(parent_query=dq).pk
    start.started = True
    start.operation = alexnet_query_by_image.name
    start.save()
    start_time = time.time()
    alexnet_query_by_image.refresh_index('alexnet')
    alexnet = alexnet_query_by_image.visual_indexer['alexnet']
    Q = entity.WQuery(dquery=dq, media_dir=settings.MEDIA_ROOT,visual_index=alexnet)
    results = Q.find(10)
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


@app.task(name="facenet_query_by_image",base=IndexerTask)
def facenet_query_by_image(query_id):
    dq = Query.objects.get(id=query_id)
    start = TEvent()
    start.video_id = Video.objects.get(parent_query=dq).pk
    start.started = True
    start.operation = facenet_query_by_image.name
    start.save()
    start_time = time.time()
    facenet_query_by_image.refresh_index('facenet')
    facenet = facenet_query_by_image.visual_indexer['facenet']
    Q = entity.WQuery(dquery=dq, media_dir=settings.MEDIA_ROOT,visual_index=facenet)
    results = Q.find(10)
    for algo,rlist in results.iteritems():
        for r in rlist:
            qr = QueryResults()
            qr.query = dq
            dd = Detection.objects.get(pk=r['detection_primary_key'])
            qr.detection = dd
            qr.frame_id = dd.frame_id
            qr.video_id = r['video_primary_key']
            qr.algorithm = algo
            qr.rank = r['rank']
            qr.distance = r['dist']
            qr.save()
    dq.results = True
    dq.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return results


def set_directory_labels(frames,dv):
    labels_to_frame = defaultdict(set)
    for f in frames:
        if f.name:
            for l in f.subdir.split('/'):
                if l.strip():
                    labels_to_frame[l].add(f.primary_key)
    for l in labels_to_frame:
        label_object, created = VLabel.objects.get_or_create(label_name=l,source=VLabel.DIRECTORY)
        _, created = VLabel.objects.get_or_create(label_name=l, source=VLabel.UI)
        for fpk in labels_to_frame[l]:
            a = Annotation()
            a.full_frame = True
            a.video = dv
            a.frame_id = fpk
            a.label_parent = label_object
            a.label = l
            a.save()

@app.task(name="extract_frames_by_id")
def extract_frames(video_id,rescale=True):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = extract_frames.name
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
    if 'RESCALE_DISABLE' in os.environ:
        rescale = False
    frames = v.extract_frames(rescale)
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
        f.primary_key = df.pk
    set_directory_labels(frames,dv)
    process_video_next(video_id,start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(name="perform_yolo_detection_by_id")
def perform_yolo_detection_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = perform_yolo_detection_by_id.name
    start.save()
    start_time = time.time()
    detector = subprocess.Popen(['fab','yolo_detect:{}'.format(video_id)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0],'../'))
    detector.wait()
    if detector.returncode != 0:
        raise ValueError,"Task failed with returncode {}".format(detector.returncode)
    process_video_next(video_id,start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(name="perform_ssd_detection_by_id")
def perform_ssd_detection_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = perform_ssd_detection_by_id.name
    start.save()
    start_time = time.time()
    detector = subprocess.Popen(['fab','ssd_detect:{}'.format(video_id)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0],'../'))
    detector.wait()
    if detector.returncode != 0:
        raise ValueError,"Task failed with returncode {}".format(detector.returncode)
    process_video_next(video_id,start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


@app.task(name="perform_face_detection_indexing_by_id")
def perform_face_detection_indexing_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = perform_face_detection_indexing_by_id.name
    start.save()
    start_time = time.time()
    face_detector = subprocess.Popen(['fab','perform_face_detection:{}'.format(video_id)],cwd=os.path.join(os.path.abspath(__file__).split('tasks.py')[0],'../'))
    face_detector.wait()
    if face_detector.returncode != 0:
        raise ValueError,"Task failed with returncode {}".format(face_detector.returncode)
    process_video_next(video_id,start.operation)
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return 0


def perform_face_indexing(video_id):
    face_indexer = indexer.FacenetIndexer()
    dv = Video.objects.get(id=video_id)
    video = entity.WVideo(dv, settings.MEDIA_ROOT)
    frames = Frame.objects.all().filter(video=dv)
    wframes = [entity.WFrame(video=video, frame_index=df.frame_index, primary_key=df.pk) for df in frames]
    input_paths = {f.local_path():f.primary_key for f in wframes}
    faces_dir = '{}/{}/detections'.format(settings.MEDIA_ROOT,video_id)
    indexes_dir = '{}/{}/indexes'.format(settings.MEDIA_ROOT,video_id)
    face_detector = detector.FaceDetector()
    aligned_paths = face_detector.detect(wframes)
    logging.info(len(aligned_paths))
    faces = []
    faces_to_pk = {}
    count = 0
    for path,v in aligned_paths.iteritems():
        for scaled_img,bb in v:
            d = Detection()
            d.video = dv
            d.confidence = 100.0
            d.frame_id = input_paths[path]
            d.object_name = "mtcnn_face"
            left, top, right, bottom = bb[0], bb[1], bb[2], bb[3]
            d.y = top
            d.x = left
            d.w = right-left
            d.h = bottom-top
            d.save()
            face_path = '{}/{}.jpg'.format(faces_dir,d.pk)
            output_filename = os.path.join(faces_dir,face_path)
            misc.imsave(output_filename, scaled_img)
            faces.append(face_path)
            faces_to_pk[face_path] = d.pk
            count += 1
    dv.refresh_from_db()
    dv.detections = dv.detections + count
    dv.save()
    path_count, emb_array, entries,feat_fname, entries_fname = face_indexer.index_faces(faces,faces_to_pk,indexes_dir,video_id)
    i = IndexEntries()
    i.video = dv
    i.count = len(entries)
    i.contains_frames = False
    i.contains_detections = True
    i.detection_name = "Face"
    i.algorithm = 'facenet'
    i.entries_file_name = entries_fname.split('/')[-1]
    i.features_file_name = feat_fname.split('/')[-1]
    i.save()


@app.task(name="export_video_by_id")
def export_video_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = export_video_by_id.name
    start.save()
    start_time = time.time()
    video_obj = Video.objects.get(pk=video_id)
    export = Export()
    export.video = video_obj
    file_name = '{}_{}.dva_export.zip'.format(video_id, int(calendar.timegm(time.gmtime())))
    export.file_name = file_name
    export.save()
    try:
        os.mkdir("{}/{}".format(settings.MEDIA_ROOT,'exports'))
    except:
        pass
    outdirname = "{}/exports/{}".format(settings.MEDIA_ROOT,video_id)
    if os.path.isdir(outdirname):
        shutil.rmtree(outdirname)
    shutil.copytree('{}/{}'.format(settings.MEDIA_ROOT,video_id),"{}/exports/{}".format(settings.MEDIA_ROOT,video_id))
    a = VideoExportSerializer(instance=video_obj)
    with file("{}/exports/{}/table_data.json".format(settings.MEDIA_ROOT,video_id),'w') as output:
        json.dump(a.data,output)
    zipper = subprocess.Popen(['zip',file_name,'-r','{}'.format(video_id)],cwd='{}/exports/'.format(settings.MEDIA_ROOT))
    zipper.wait()
    if zipper.returncode != 0:
        raise ValueError,"Could not zip {}".format(zipper.returncode)
    shutil.rmtree("{}/exports/{}".format(settings.MEDIA_ROOT,video_id))
    export.completed = True
    export.save()
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()
    return zipper.returncode


@app.task(name="import_video_by_id")
def import_video_by_id(video_id):
    start = TEvent()
    start.video_id = video_id
    start.started = True
    start.operation = import_video_by_id.name
    start.save()
    start_time = time.time()
    video_obj = Video.objects.get(pk=video_id)
    zipf = zipfile.ZipFile("{}/{}/{}.zip".format(settings.MEDIA_ROOT, video_id, video_id), 'r')
    zipf.extractall("{}/{}/".format(settings.MEDIA_ROOT, video_id))
    zipf.close()
    video_root_dir = "{}/{}/".format(settings.MEDIA_ROOT, video_id)
    old_key = None
    for k in os.listdir(video_root_dir):
        unzipped_dir = "{}{}".format(video_root_dir, k)
        old_key = k
        if os.path.isdir(unzipped_dir):
            for subdir in os.listdir(unzipped_dir):
                shutil.move("{}/{}".format(unzipped_dir,subdir),"{}".format(video_root_dir))
            shutil.rmtree(unzipped_dir)
            break
    with open("{}/{}/table_data.json".format(settings.MEDIA_ROOT, video_id)) as input_json:
        video_json = json.load(input_json)
    video_obj.name = 'Imported:'+ video_json['name']
    video_obj.frames = video_json['frames']
    video_obj.detections = video_json['detections']
    video_obj.youtube_video = video_json['youtube_video']
    video_obj.dataset = video_json['dataset']
    video_obj.url = video_json['url']
    video_obj.description = video_json['description']
    video_obj.metadata = video_json['metadata']
    video_obj.length_in_seconds = video_json['length_in_seconds']
    video_obj.save()
    if not video_obj.dataset:
        os.rename("{}/video/{}.mp4".format(video_root_dir,old_key),"{}/video/{}.mp4".format(video_root_dir,video_id))
    frame_to_pk = {}
    detection_to_pk = {}
    for f in video_json['frame_list']:
        df = Frame()
        df.video = video_obj
        df.name = f['name']
        df.frame_index = f['frame_index']
        df.subdir = f['subdir']
        df.save()
        frame_to_pk[f['id']]=df.pk
    for d in video_json['detection_list']:
        dd = Detection()
        dd.video = video_obj
        dd.x = d['x']
        dd.y = d['y']
        dd.h = d['h']
        dd.w = d['w']
        dd.frame_id = frame_to_pk[d['frame']]
        dd.confidence = d['confidence']
        dd.object_name = d['object_name']
        dd.metadata = d['metadata']
        dd.save()
        detection_to_pk[d['id']]=dd.pk
    for k,v in detection_to_pk.iteritems():
        os.rename('{}/detections/{}.jpg'.format(video_root_dir,k),"{}/detections/d_{}.jpg".format(video_root_dir,v))
    # this is done to avoid accidental overlap when renaming files.
    for v in detection_to_pk.itervalues():
        os.rename('{}/detections/d_{}.jpg'.format(video_root_dir,v),"{}/detections/{}.jpg".format(video_root_dir,v))
    for i in video_json['index_entries_list']:
        di = IndexEntries()
        di.video = video_obj
        di.algorithm = i['algorithm']
        di.count = i['count']
        di.contains_detections = i['contains_detections']
        di.contains_frames = i['contains_frames']
        di.approximate = i['approximate']
        di.created = i['created']
        di.features_file_name = i['features_file_name']
        di.entries_file_name = i['entries_file_name']
        di.detection_name = i['detection_name']
        di.save()
    os.remove("{}/{}/{}.zip".format(settings.MEDIA_ROOT, video_id, video_id))
    start.completed = True
    start.seconds = time.time() - start_time
    start.save()

