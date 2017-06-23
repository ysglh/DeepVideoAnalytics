import shlex,json,os,zipfile,glob,logging
import subprocess as sp
import numpy as np
from PIL import Image
import os
from collections import defaultdict
from ..models import Video,Frame,Segment,Scene,AppliedLabel


def set_directory_labels(frames, dv):
    labels_to_frame = defaultdict(set)
    for f in frames:
        if f.name:
            for l in f.subdir.split('/')[1:]:
                if l.strip():
                    labels_to_frame[l].add(f.primary_key)
    label_list = []
    for l in labels_to_frame:
        for fpk in labels_to_frame[l]:
            a = AppliedLabel()
            a.video = dv
            a.frame_id = fpk
            a.source = AppliedLabel.DIRECTORY
            a.label_name = l
            label_list.append(a)
    AppliedLabel.objects.bulk_create(label_list)



class WVideo(object):
    """
    Wrapper object for a video / dataset
    """

    def __init__(self,dvideo,media_dir):
        self.dvideo = dvideo
        self.primary_key = self.dvideo.pk
        self.media_dir = media_dir
        self.local_path = "{}/{}/video/{}.mp4".format(self.media_dir,self.primary_key,self.primary_key)
        self.duration = None
        self.width = None
        self.height = None
        self.metadata = {}

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


    def index_frames(self,frames,visual_index):
        results = {}
        wframes = [WFrame(video=self, frame_index=df.frame_index,primary_key=df.pk) for df in frames]
        visual_index.load()
        entries = []
        paths = []
        for i, f in enumerate(wframes):
            entry = {
                'frame_index': f.frame_index,
                'frame_primary_key': f.primary_key,
                'video_primary_key': self.primary_key,
                'index': i,
                'type': 'frame'
            }
            paths.append(f.local_path())
            entries.append(entry)
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/frames_{}.npy".format(self.media_dir, self.primary_key, visual_index.name)
        entries_fname = "{}/{}/indexes/frames_{}.json".format(self.media_dir, self.primary_key, visual_index.name)
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
            path = "{}/{}/detections/{}.jpg".format(self.media_dir, self.primary_key, d.pk)
            if os.path.isfile(path=path):
                paths.append(path)
            else:
                img = Image.open("{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, d.frame.frame_index))
                img2 = img.crop((d.x, d.y, d.x+d.w, d.y+d.h))
                img2.save(path)
                paths.append(path)
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
        if not args['perform_scene_detection']:
            logging.warning("Scene detection is disabled")
        if args['rate']:
            denominator = int(args['rate'])
        else:
            denominator = 30
        rescale = args['rescale'] if 'rescale' in args else 0
        if self.dvideo.dataset:
            self.extract_zip_dataset()
        else:
            self.get_metadata()
            self.extract_video_frames(denominator,rescale)

    def extract_video_frames(self,denominator,rescale):
        output_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'frames')
        ffmpeg_command = 'ffmpeg -i {} -vf'.format(self.local_path)
        if rescale:
            filter_command_denominator = '"select=not(mod(n\,{})),scale={}:-1" -vsync 0'.format(denominator,rescale)
            filter_command_keyframe = '"select=eq(pict_type\,PICT_TYPE_I),scale={}:-1" -vsync 0'.format(rescale)
        else:
            filter_command_denominator = '"select=not(mod(n\,{}))" -vsync 0'.format(denominator)
            filter_command_keyframe = '"select=eq(pict_type\,PICT_TYPE_I)" -vsync 0'.format()
        output_command = "{}/%d_b.jpg".format(output_dir)
        denominator_command = " ".join([ffmpeg_command,filter_command_denominator,output_command])
        output_command = "{}/%d_k.jpg -loglevel debug".format(output_dir)
        keyframe_command = " ".join([ffmpeg_command, filter_command_keyframe, output_command])
        logging.info(denominator_command)
        _ = sp.check_output(shlex.split(denominator_command),stderr=sp.STDOUT)
        logging.info(keyframe_command)
        keyframes_info = sp.check_output(shlex.split(keyframe_command),stderr=sp.STDOUT)
        with open("{}{}".format(output_dir,"keyframes.txt"),'w') as out:
            out.write(keyframes_info)
        for fname in glob.glob(output_dir + '*_b.jpg'):
            ind = int(fname.split('/')[-1].replace('_b.jpg', ''))
            os.rename(fname, fname.replace('{}_b.jpg'.format(ind), '{}.jpg'.format((ind - 1) * denominator)))
        frame_width, frame_height = 0, 0
        df_list = []
        for i, fname in enumerate(glob.glob(output_dir + '*.jpg')):
            if i == 0:
                im = Image.open(fname)
                frame_width, frame_height = im.size  # this remains constant for all frames
            if not fname.endswith('_k.jpg'):
                frame_name = fname.split('/')[-1].split('.')[0]
                ind = int(frame_name)
                df = Frame()
                df.frame_index = int(ind)
                df.video_id = self.dvideo.pk
                df.h = frame_height
                df.w = frame_width
                df_list.append(df)
        df_ids = Frame.objects.bulk_create(df_list)
        self.dvideo.frames = len(df_list)
        index_to_df = {}
        self.dvideo.save()
        df_list = []
        for i, k in enumerate(df_ids):
            index_to_df[df_list[i].frame_index] = k.id

    def segment_video(self):
        segments = []
        segments_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'segments')
        command = 'ffmpeg -i {} -c copy -map 0 -segment_time 1 -f segment -reset_timestamps 1 ' \
                  '-segment_list_type csv -segment_list {}/segments.csv ' \
                  '{}/%d.mp4'.format(self.local_path, segments_dir, segments_dir)
        logging.info(command)
        segmentor = sp.Popen(shlex.split(command))
        segmentor.wait()
        if segmentor.returncode != 0:
            raise ValueError
        else:
            for line in file('{}/segments.csv'.format(segments_dir)):
                segment_file_name, start_time, end_time = line.strip().split(',')
                command = 'ffprobe -select_streams v -show_streams -show_frames -print_format json {}  '.format(segment_file_name)
                logging.info(command)
                segment_json = sp.check_output(shlex.split(command), cwd=segments_dir)
                segments.append((int(segment_file_name.split('.')[0]), float(start_time), float(end_time), segment_json))
            segments.sort()
        for s in segments:
            segment_id, start_time, end_time, metadata = s
            ds = Segment()
            ds.segment_index = segment_id
            ds.start_time = start_time
            ds.end_time = end_time
            ds.video_id = self.dvideo.pk
            ds.metadata = metadata
            metadata_json = json.loads(metadata)
            ds.frame_count = int(metadata_json["streams"][0]["nb_frames"])
            ds.save()


    def detect_scenes(self,index_to_df,rescale,start):
        cwd = os.path.join(os.path.abspath(__file__).split('entity.py')[0], '../')
        scencedetect = sp.Popen(['fab', 'pyscenedetect:{},{}'.format(self.primary_key, rescale)], cwd=cwd)
        scencedetect.wait()
        if scencedetect.returncode != 0:
            logging.info("pyscene detect failed with {} check fab.log for the reason".format(scencedetect.returncode))
        else:
            with open('{}/{}/frames/scenes.json'.format(self.media_dir, self.primary_key)) as fh:
                cuts = json.load(fh)
            os.remove('{}/{}/frames/scenes.json'.format(self.media_dir, self.primary_key))
        cust_list = [(cuts[cutindex], cuts[cutindex + 1]) for cutindex, cut in enumerate(sorted(cuts)[:-1])]
        for start_frame_index, end_frame_index in cust_list:
            ds = Scene()
            ds.video = self.dvideo
            ds.start_frame_index = start_frame_index
            ds.end_frame_index = end_frame_index
            ds.start_frame_id = index_to_df[start_frame_index]
            ds.end_frame_id = index_to_df[end_frame_index]
            ds.source = start
            ds.save()

    def extract_zip_dataset(self):
        frames = []
        zipf = zipfile.ZipFile("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key), 'r')
        zipf.extractall("{}/{}/frames/".format(self.media_dir, self.primary_key))
        zipf.close()
        i = 0
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
                            f = WFrame(frame_index=i, video=self, name=fname.split('/')[-1],
                                       subdir=subdir.replace("{}/{}/frames/".format(self.media_dir, self.primary_key),
                                                             ''), w=w, h=h)
                            frames.append(f)
                    else:
                        logging.warning("skipping {} not a jpeg file".format(fname))
            else:
                logging.warning("skipping {} ".format(subdir))
        self.dvideo.frames = len(frames)
        self.dvideo.save()
        df_list = []
        for f in frames:
            df = Frame()
            df.frame_index = f.frame_index
            df.video_id = self.dvideo.pk
            if f.h:
                df.h = f.h
            if f.w:
                df.w = f.w
            if f.name:
                df.name = f.name[:150]
                df.subdir = f.subdir.replace('/', ' ')
            df_list.append(df)
        df_ids = Frame.objects.bulk_create(df_list)
        set_directory_labels(frames, self.dvideo)


class WFrame(object):

    def __init__(self,frame_index=None,video=None,primary_key=None,name=None,subdir=None,h=None,w=None):
        if video:
            self.subdir = subdir
            self.frame_index = frame_index
            self.video = video
            self.primary_key = primary_key
            self.name = name
        else:
            self.subdir = None
            self.frame_index = None
            self.video = None
            self.primary_key = None
            self.name = None
        self.h = h
        self.w = w

    def local_path(self):
        return "{}/{}/{}/{}.jpg".format(self.video.media_dir,self.video.primary_key,'frames',self.frame_index)

