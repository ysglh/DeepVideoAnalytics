import base64, time
import shlex,json,os,zipfile,glob,logging
import subprocess as sp
import boto3
from django.conf import settings
from dva.celery import app
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from PIL import Image


from ..models import Video,Query,IndexerQuery,QueryResults,Region,ClusterCodes,TEvent,Frame,Segment,Tube,AppliedLabel
from collections import defaultdict
from celery.result import AsyncResult
import io


def query_approximate(local_path, n, visual_index, clusterer):
    vector = visual_index.apply(local_path)
    results = []
    coarse, fine, results_indexes = clusterer.apply(vector, n)
    for i, k in enumerate(results_indexes[0]):
        e = ClusterCodes.objects.get(searcher_index=k.id, clusters=clusterer.dc)
        if e.detection_id:
            results.append({
                'rank': i + 1,
                'dist': i,
                'detection_primary_key': e.detection_id,
                'frame_index': e.frame.frame_index,
                'frame_primary_key': e.frame_id,
                'video_primary_key': e.video_id,
                'type': 'detection',
            })
        else:
            results.append({
                'rank': i + 1,
                'dist': i,
                'frame_index': e.frame.frame_index,
                'frame_primary_key': e.frame_id,
                'video_primary_key': e.video_id,
                'type': 'frame',
            })
    return results


class DVAPQLProcess(object):

    def __init__(self):
        self.query = None
        self.media_dir = None
        self.indexer_queries = []
        self.task_results = {}
        self.context = {}
        self.dv = None
        self.visual_indexes = settings.VISUAL_INDEXES

    def store_and_create_video_object(self):
        self.dv = Video()
        self.dv.name = 'query_{}'.format(self.query.pk)
        self.dv.dataset = True
        self.dv.query = True
        self.dv.parent_query = self.query
        self.dv.save()
        if settings.HEROKU_DEPLOY:
            query_key = "queries/{}.png".format(self.query.pk)
            query_frame_key = "{}/frames/0.png".format(self.dv.pk)
            s3 = boto3.resource('s3')
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_key, Body=self.query.image_data)
            s3.Bucket(settings.MEDIA_BUCKET).put_object(Key=query_frame_key, Body=self.query.image_data)
        else:
            query_path = "{}/queries/{}.png".format(settings.MEDIA_ROOT, self.query.pk)
            with open(query_path, 'w') as fh:
                fh.write(self.query.image_data)

    def create_from_request(self, request):
        count = request.POST.get('count')
        excluded_index_entries_pk = json.loads(request.POST.get('excluded_index_entries'))
        selected_indexers = json.loads(request.POST.get('selected_indexers'))
        approximate = True if request.POST.get('approximate') == 'true' else False
        image_data_url = request.POST.get('image_url')
        user = request.user if request.user.is_authenticated else None
        self.query = Query()
        self.query.approximate = approximate
        if not (user is None):
            self.query.user = user
        image_data = base64.decodestring(image_data_url[22:])
        self.query.image_data = image_data
        self.query.save()
        self.store_and_create_video_object()
        for k in selected_indexers:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k
            iq.count = count
            if excluded_index_entries_pk:
                # !!fix this only the indexer specific
                iq.excluded_index_entries_pk = [int(epk) for epk in excluded_index_entries_pk]
            iq.approximate = approximate
            iq.save()
            self.indexer_queries.append(iq)
        return self.query

    def create_from_json(self, j, user=None):
        """
        Create query from JSON
        {
        'image_data':base64.encodestring(file('tests/query.png').read()),
        'indexers':[
            {
                'algorithm':'facenet',
                'count':10,
                'approximate':False
            }
            ]
        }
        :param j: JSON encoded query
        :param user:
        :return:
        """
        self.query = Query()
        if not (user is None):
            self.query.user = user
        if j['image_data_b64'].strip():
            image_data = base64.decodestring(j['image_data_b64'])
            self.query.image_data = image_data
        self.query.save()
        self.store_and_create_video_object()
        for k in j['indexers']:
            iq = IndexerQuery()
            iq.parent_query = self.query
            iq.algorithm = k['algorithm']
            iq.count = k['count']
            iq.excluded_index_entries_pk = k['excluded_index_entries_pk'] if 'excluded_index_entries_pk' in k else []
            iq.approximate = k['approximate']
            iq.save()
            self.indexer_queries.append(iq)
        return self.query

    def send_tasks(self):
        for iq in self.indexer_queries:
            task_name = 'perform_indexing'
            queue_name = self.visual_indexes[iq.algorithm]['indexer_queue']
            jargs = json.dumps({
                'iq_id':iq.pk,
                'index':iq.algorithm,
                'target':'query',
                'next_tasks':[
                    { 'task_name': 'perform_retrieval',
                      'arguments': {'iq_id': iq.pk,'index':iq.algorithm}
                     }
                ]
            })
            next_task = TEvent.objects.create(video=self.dv, operation=task_name, arguments_json=jargs)
            self.task_results[iq.algorithm] = app.send_task(task_name, args=[next_task.pk, ], queue=queue_name, priority=5)
            self.context[iq.algorithm] = []

    def wait(self,timeout=60):
        for visual_index_name, result in self.task_results.iteritems():
            try:
                next_task_ids = result.get(timeout=timeout)
                if next_task_ids:
                    for next_task_id in next_task_ids:
                        next_result = AsyncResult(id=next_task_id)
                        _ = next_result.get(timeout=timeout)
            except Exception, e:
                raise ValueError(e)

    def collect_results(self):
        self.context = defaultdict(list)
        for r in QueryResults.objects.all().filter(query=self.query):
            self.context[r.algorithm].append((r.rank,
                                         {'url': '{}{}/regions/{}.jpg'.format(settings.MEDIA_URL, r.video_id,
                                                                                 r.detection_id) if r.detection_id else '{}{}/frames/{}.jpg'.format(
                                             settings.MEDIA_URL, r.video_id, r.frame.frame_index),
                                          'result_type': "Region" if r.detection_id else "Frame",
                                          'rank':r.rank,
                                          'frame_id': r.frame_id,
                                          'frame_index': r.frame.frame_index,
                                          'distance': r.distance,
                                          'video_id': r.video_id,
                                          'video_name': r.video.name}))
        for k, v in self.context.iteritems():
            if v:
                self.context[k].sort()
                self.context[k] = zip(*v)[1]

    def load_from_db(self,query,media_dir):
        self.query = query
        self.media_dir = media_dir

    def to_json(self):
        json_query = {
        }
        return json.dumps(json_query)

    def execute_sub_query(self,iq,visual_index):
        """
        TODO move this inside indexing task
        :param iq:
        :param visual_index:
        :return:
        """
        local_path = "{}/queries/{}_{}.png".format(self.media_dir, iq.algorithm, self.query.pk)
        with open(local_path, 'w') as fh:
            fh.write(str(self.query.image_data))
        vector = visual_index.apply(local_path)
        s = io.BytesIO()
        np.save(s,vector) # TODO: figure out a better way to store numpy arrays.
        iq.vector = s.getvalue()
        iq.save()
        self.query.results_available = True
        self.query.save()
        return 0

    def perform_retrieval(self,iq,index_name,retrieval_task):
        """
        TODO move this inside retrival task
        :param iq:
        :param index_name:
        :param retrieval_task:
        :return:
        """
        retriever = retrieval_task.visual_retriever[index_name]
        exact = True
        results = []
        vector = np.load(io.BytesIO(iq.vector)) # TODO: figure out a better way to store numpy arrays.
        if iq.approximate:
            if retrieval_task.clusterer[index_name] is None:
                retrieval_task.load_clusterer(index_name)
            clusterer = retrieval_task.clusterer[index_name]
            if clusterer:
                results = query_approximate(retrieval_task, iq.count, retriever, clusterer)
                exact = False
        if exact:
            retrieval_task.refresh_index(index_name)
            results = retriever.nearest(vector=vector,n=iq.count)
        # TODO: optimize this using batching
        for r in results:
            qr = QueryResults()
            qr.query = self.query
            qr.indexerquery = iq
            if 'detection_primary_key' in r:
                dd = Region.objects.get(pk=r['detection_primary_key'])
                qr.detection = dd
                qr.frame_id = dd.frame_id
            else:
                qr.frame_id = r['frame_primary_key']
            qr.video_id = r['video_primary_key']
            qr.algorithm = iq.algorithm
            qr.rank = r['rank']
            qr.distance = r['dist']
            qr.save()
        iq.results = True
        iq.save()
        self.query.results_available = True
        self.query.save()
        return 0


class WVideo(object):
    """
    Wrapper object for a video / dataset
    """

    def __init__(self,dvideo,media_dir):
        self.dvideo = dvideo
        self.primary_key = self.dvideo.pk
        self.media_dir = media_dir
        self.local_path = "{}/{}/video/{}.mp4".format(self.media_dir,self.primary_key,self.primary_key)
        self.segments_dir = "{}/{}/segments/".format(self.media_dir,self.primary_key)
        self.duration = None
        self.width = None
        self.height = None
        self.metadata = {}
        self.segment_frames_dict = {}
        self.csv_format = None

    def detect_csv_segment_format(self):
        format_path = "{}format.txt".format(self.segments_dir)
        self.csv_format = {}
        if not os.path.isfile(format_path):
            command ="ffprobe -i {}0.mp4 -show_frames -select_streams v:0 -print_format csv=nokey=0".format(self.segments_dir)
            csv_format_lines = sp.check_output(shlex.split(command))
            with open(format_path,'w') as formatfile:
                formatfile.write(csv_format_lines)
            logging.info("Generated csv format {}".format(self.csv_format))
        for line in file(format_path).read().splitlines():
            if line.strip():
                for i,kv in enumerate(line.strip().split(',')):
                    if '=' in kv:
                        k,v = kv.strip().split('=')
                        self.csv_format[k] = i
                    else:
                        self.csv_format[kv] = i
                break
        self.field_count = len(self.csv_format)
        self.pict_type_index = self.csv_format['pict_type']
        self.time_index = self.csv_format['best_effort_timestamp_time']

    def parse_segment_framelist(self,segment_id, framelist):
        if self.csv_format is None:
            self.detect_csv_segment_format()
        frames = {}
        findex = 0
        for line in framelist.splitlines():
            if line.strip():
                entries = line.strip().split(',')
                if len(entries) == self.field_count:
                    frames[findex] = {'type': entries[self.pict_type_index], 'ts': float(entries[self.time_index])}
                    findex += 1
                else:
                    errro_message = "format used {} \n {} (expected) != {} entries in {} \n {} ".format(self.csv_format,self.field_count,len(entries),segment_id, line)
                    logging.error(errro_message)
                    raise ValueError, errro_message
        return frames

    def get_metadata(self):
        if self.dvideo.youtube_video:
            output_dir = "{}/{}/{}/".format(self.media_dir,self.primary_key,'video')
            command = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'  \"{}\" -o {}.mp4".format(self.dvideo.url,self.primary_key)
            logging.info(command)
            download = sp.Popen(shlex.split(command),cwd=output_dir)
            download.wait()
            if download.returncode != 0:
                logging.error("Could not download the video")
                raise ValueError
        command = ['ffprobe','-i',self.local_path,'-print_format','json','-show_format','-show_streams','-v','quiet']
        p = sp.Popen(command,stdout=sp.PIPE,stderr=sp.STDOUT,stdin=sp.PIPE)
        p.wait()
        output, _ = p.communicate()
        self.metadata = json.loads(output)
        try:
            self.duration = float(self.metadata['format']['duration'])
            self.width = float(self.metadata['streams'][0]['width'])
            self.height = float(self.metadata['streams'][0]['height'])
        except:
            raise ValueError,str(self.metadata)
        self.dvideo.metadata = self.metadata
        self.dvideo.length_in_seconds = self.duration
        self.dvideo.height = self.height
        self.dvideo.width = self.width
        self.dvideo.save()

    def index_frames(self,frames,visual_index,task_pk):
        visual_index.load()
        entries = []
        paths = []
        for i, df in enumerate(frames):
            entry = {
                'frame_index': df.frame_index,
                'frame_primary_key': df.pk,
                'video_primary_key': self.primary_key,
                'index': i,
                'type': 'frame'
            }
            paths.append("{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, df.frame_index))
            entries.append(entry)
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/frames_{}_{}.npy".format(self.media_dir, self.primary_key, visual_index.name,task_pk)
        entries_fname = "{}/{}/indexes/frames_{}_{}.json".format(self.media_dir, self.primary_key, visual_index.name,task_pk)
        with open(feat_fname, 'w') as feats:
            np.save(feats, np.array(features))
        with open(entries_fname, 'w') as entryfile:
            json.dump(entries, entryfile)
        return visual_index.name,entries,feat_fname,entries_fname

    def index_regions(self,regions,regions_name,visual_index):
        visual_index.load()
        entries = []
        paths = []
        for i, d in enumerate(regions):
            entry = {
                'frame_index': d.frame.frame_index,
                'detection_primary_key': d.pk,
                'frame_primary_key': d.frame.pk,
                'video_primary_key': self.primary_key,
                'index': i,
                'type': d.region_type
            }
            path = "{}/{}/regions/{}.jpg".format(self.media_dir, self.primary_key, d.pk)
            if d.materialized:
                paths.append(path)
            else:
                img = Image.open("{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, d.frame.frame_index))
                img2 = img.crop((d.x, d.y, d.x+d.w, d.y+d.h))
                img2.save(path)
                paths.append(path)
                d.materialized = True
                d.save()
            entries.append(entry)
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/{}_{}.npy".format(self.media_dir, self.primary_key,regions_name, visual_index.name)
        entries_fname = "{}/{}/indexes/{}_{}.json".format(self.media_dir, self.primary_key,regions_name, visual_index.name)
        with open(feat_fname, 'w') as feats:
            np.save(feats, np.array(features))
        with open(entries_fname, 'w') as entryfile:
            json.dump(entries, entryfile)
        return visual_index.name,entries,feat_fname,entries_fname

    def extract(self,args,start):
        self.extract_zip_dataset()

    def decode_segment(self,ds,denominator,rescale):
        output_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'frames')
        segments_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'segments',)
        input_segment = "{}{}.mp4".format(self.segments_dir, ds.segment_index)
        ffmpeg_command = 'ffmpeg -fflags +igndts -loglevel panic -i {} -vf'.format(input_segment) # Alternative to igndts is setting vsync vfr
        df_list = []
        if rescale:
            filter_command = '"select=not(mod(n\,{}))+eq(pict_type\,PICT_TYPE_I),scale={}:-1" -vsync 0'.format(denominator,rescale)
        else:
            filter_command = '"select=not(mod(n\,{}))+eq(pict_type\,PICT_TYPE_I)" -vsync 0'.format(denominator)
        output_command = "{}/segment_{}_%d_b.jpg".format(output_dir,ds.segment_index)
        command = " ".join([ffmpeg_command,filter_command,output_command])
        logging.info(command)
        try:
            _ = sp.check_output(shlex.split(command), stderr=sp.STDOUT)
        except:
            raise ValueError,"for {} could not run {}".format(self.dvideo.name,command)
        with open("{}{}.txt".format(segments_dir, ds.segment_index)) as framelist:
            segment_frames_dict = self.parse_segment_framelist(ds.segment_index, framelist.read())
        ordered_frames = sorted([(k,v) for k,v in segment_frames_dict.iteritems() if k%denominator == 0 or v['type'] == 'I'])
        frame_width, frame_height = 0, 0
        for i,f_id in enumerate(ordered_frames):
            frame_index, frame_data = f_id
            src = "{}/segment_{}_{}_b.jpg".format(output_dir,ds.segment_index,i+1)
            dst = "{}/{}.jpg".format(output_dir,frame_index+ds.start_index)
            try:
                os.rename(src,dst)
            except:
                raise ValueError,str((src, dst, frame_index, ds.start_index))
            if i ==0:
                im = Image.open(dst)
                frame_width, frame_height = im.size  # this remains constant for all frames
            df = Frame()
            df.frame_index = int(frame_index+ds.start_index)
            df.video_id = self.dvideo.pk
            df.keyframe = True if frame_data['type'] == 'I' else False
            df.t = frame_data['ts']
            df.segment_index = ds.segment_index
            df.h = frame_height
            df.w = frame_width
            df_list.append(df)
        _ = Frame.objects.bulk_create(df_list, batch_size=1000)

    def segment_video(self):
        segments_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'segments')
        command = 'ffmpeg -loglevel panic -i {} -c copy -map 0 -segment_time 1 -f segment ' \
                  '-segment_list_type csv -segment_list {}/segments.csv ' \
                  '{}/%d.mp4'.format(self.local_path, segments_dir, segments_dir)
        logging.info(command)
        segmentor = sp.Popen(shlex.split(command))
        segmentor.wait()
        if segmentor.returncode != 0:
            raise ValueError
        else:
            timer_start = time.time()
            start_index = 0
            for line in file('{}/segments.csv'.format(segments_dir)):
                segment_file_name, start_time, end_time = line.strip().split(',')
                segment_id = int(segment_file_name.split('.')[0])
                command = 'ffprobe -select_streams v -show_streams  -print_format json {}  '.format(segment_file_name)
                # logging.info(command)
                segment_json = sp.check_output(shlex.split(command), cwd=segments_dir)
                command = 'ffprobe -show_frames -select_streams v:0 -print_format csv {}'.format(segment_file_name)
                # logging.info(command)
                framelist= sp.check_output(shlex.split(command), cwd=segments_dir)
                with open("{}/{}.txt".format(segments_dir,segment_file_name.split('.')[0]),'w') as framesout:
                    framesout.write(framelist)
                self.segment_frames_dict[segment_id] = self.parse_segment_framelist(segment_id,framelist)
                logging.warning("Processing line {}".format(line))
                start_time = float(start_time)
                end_time = float(end_time)
                ds = Segment()
                ds.segment_index = segment_id
                ds.start_time = start_time
                ds.start_index = start_index
                start_index += len(self.segment_frames_dict[segment_id])
                ds.frame_count = len(self.segment_frames_dict[segment_id])
                ds.end_time = end_time
                ds.video_id = self.dvideo.pk
                ds.metadata = segment_json
                ds.save()
            logging.info("Took {} seconds to process {} segments".format(time.time() - timer_start,len(self.segment_frames_dict)))
        self.dvideo.frames = sum([len(c) for c in self.segment_frames_dict.itervalues()])
        self.dvideo.segments = len(self.segment_frames_dict)
        self.dvideo.save()
        self.detect_csv_segment_format() # detect and save

    def extract_zip_dataset(self):
        zipf = zipfile.ZipFile("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key), 'r')
        zipf.extractall("{}/{}/frames/".format(self.media_dir, self.primary_key))
        zipf.close()
        i = 0
        df_list = []
        root_length = len("{}/{}/frames/".format(self.media_dir, self.primary_key))
        for subdir, dirs, files in os.walk("{}/{}/frames/".format(self.media_dir, self.primary_key)):
            if '__MACOSX' not in subdir:
                for fname in files:
                    fname = os.path.join(subdir, fname)
                    if fname.endswith('jpg') or fname.endswith('jpeg'):
                        i += 1
                        try:
                            im = Image.open(fname)
                            w, h = im.size
                        except IOError:
                            logging.info("Could not open {} skipping".format(fname))
                        else:
                            dst = "{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, i)
                            os.rename(fname, dst)
                            df = Frame()
                            df.frame_index = i
                            df.video_id = self.dvideo.pk
                            df.h = h
                            df.w = w
                            df.name = fname.split('/')[-1][:150]
                            df.subdir = subdir[root_length:].replace('/', ' ')
                            df_list.append(df)

                    else:
                        logging.warning("skipping {} not a jpeg file".format(fname))
            else:
                logging.warning("skipping {} ".format(subdir))
        self.dvideo.frames = len(df_list)
        self.dvideo.save()
        df_ids = Frame.objects.bulk_create(df_list,batch_size=1000)
        labels_to_frame = defaultdict(set)
        for i,f in enumerate(df_list):
            if f.name:
                for l in f.subdir.split(' ')[1:]:
                    if l.strip():
                        labels_to_frame[l].add(df_ids[i].id)
        label_list = []
        for l in labels_to_frame:
            for fpk in labels_to_frame[l]:
                a = AppliedLabel()
                a.video_id = self.dvideo.pk
                a.frame_id = fpk
                a.source = AppliedLabel.DIRECTORY
                a.label_name = l
                label_list.append(a)
        AppliedLabel.objects.bulk_create(label_list, batch_size=1000)
