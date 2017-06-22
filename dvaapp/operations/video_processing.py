import shlex,json,os,zipfile,glob,logging
import subprocess as sp
import numpy as np
from PIL import Image
import os
from collections import defaultdict


class WVideo(object):
    """
    Wrapper object for a video
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

    def perform_video_processing(self,args,frames,cuts,segments,key_frames):
        if args['rate']:
            denominator = int(args['rate'])
        else:
            denominator = 30

        output_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'frames')
        if args['rescale']:
            command = 'ffmpeg -i {} -vf "select=not(mod(n\,{})),scale={}:-1" -vsync 0  {}/%d_b.jpg'.format(
                self.local_path, denominator, int(args['rescale']), output_dir)
            kf_commmand = 'ffmpeg -i {} -vf "select=eq(pict_type\,PICT_TYPE_I),scale={}:-1" -vsync 0 {}/k_%d.jpg -loglevel debug'.format(
                self.local_path, int(args['rescale']), output_dir)
        else:
            command = 'ffmpeg -i {} -vf "select=not(mod(n\,{}))" -vsync 0  {}/%d_b.jpg'.format(self.local_path,
                                                                                               denominator, output_dir)
            kf_commmand = 'ffmpeg -i {} -vf "select=eq(pict_type\,PICT_TYPE_I)" -vsync 0 {}/k_%d.jpg -loglevel debug'.format(
                self.local_path, output_dir)
        extract = sp.Popen(shlex.split(command))
        extract.wait()
        key_frame_extract = sp.check_output(shlex.split(kf_commmand), stderr=sp.STDOUT)
        count = None
        for line in key_frame_extract.split('\n'):
            if "pict_type:I" in line:
                if count is None:
                    count = 0
                for l in line.strip().split(' '):
                    if l.startswith('n:') or l.startswith('t:'):
                        ka, va = l.split(':')
                        if ka == 'n':
                            if int(float(va)) != count:
                                logging.warning("{} != {} is not frame index".format(va,count))
                        key_frames[count][ka] = va
                src, dst = '{}/k_{}.jpg'.format(output_dir, len(key_frames)), '{}/{}.jpg'.format(output_dir, count)
                os.rename(src, dst)
            if "pict_type" in line and not (count is None):
                count += 1
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
                command = 'ffprobe -select_streams v -show_streams -print_format json {}  '.format(
                    segment_file_name)  # -show_frames for frame specific metadata
                logging.info(command)
                segment_json = sp.check_output(shlex.split(command), cwd=segments_dir)
                segments.append(
                    (int(segment_file_name.split('.')[0]), float(start_time), float(end_time), segment_json))
            segments.sort()
        for fname in glob.glob(output_dir + '*_b.jpg'):
            ind = int(fname.split('/')[-1].replace('_b.jpg', ''))
            os.rename(fname, fname.replace('{}_b.jpg'.format(ind), '{}.jpg'.format((ind - 1) * denominator)))
        if not args['perform_scene_detection']:
            logging.warning("Scene detection is disabled")
        else:
            if not args['rescale']:
                args['rescale'] = 0
            scencedetect = sp.Popen(['fab', 'pyscenedetect:{},{}'.format(self.primary_key, args['rescale'])],
                                    cwd=os.path.join(os.path.abspath(__file__).split('entity.py')[0], '../'))
            scencedetect.wait()
            if scencedetect.returncode != 0:
                logging.info(
                    "pyscene detect failed with {} check fab.log for the reason".format(scencedetect.returncode))
            else:
                with open('{}/{}/frames/scenes.json'.format(self.media_dir, self.primary_key)) as fh:
                    cuts = json.load(fh)
                os.remove('{}/{}/frames/scenes.json'.format(self.media_dir, self.primary_key))
        frame_width, frame_height = 0, 0
        for i, fname in enumerate(glob.glob(output_dir + '*.jpg')):
            frame_name = fname.split('/')[-1].split('.')[0]
            if not frame_name.startswith('k'):
                ind = int(frame_name)
                if i == 0:
                    im = Image.open(fname)
                    frame_width, frame_height = im.size  # this remains constant for all frames
                f = WFrame(frame_index=int(ind), video=self, w=frame_width, h=frame_height)
                frames.append(f)

    def extract_frames(self,args):
        frames = []
        cuts = []
        segments = []
        key_frames = defaultdict(dict)
        if not self.dvideo.dataset:
            self.perform_video_processing(args,frames=frames,cuts=cuts,segments=segments,key_frames=key_frames)
        else:
            zipf = zipfile.ZipFile("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key), 'r')
            zipf.extractall("{}/{}/frames/".format(self.media_dir, self.primary_key))
            zipf.close()
            i = 0
            for subdir, dirs, files in os.walk("{}/{}/frames/".format(self.media_dir, self.primary_key)):
                if '__MACOSX' not in subdir:
                    for fname in files:
                        fname = os.path.join(subdir,fname)
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
                                f = WFrame(frame_index=i, video=self,name=fname.split('/')[-1],subdir=subdir.replace("{}/{}/frames/".format(self.media_dir, self.primary_key),''),w=w,h=h)
                                frames.append(f)
                        else:
                            logging.warning("skipping {} not a jpeg file".format(fname))
                else:
                    logging.warning("skipping {} ".format(subdir))
        return frames, cuts, segments, key_frames

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

