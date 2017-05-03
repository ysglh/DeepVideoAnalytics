import shlex,json,os,zipfile,glob,logging
import subprocess as sp
import numpy as np
import pyscenecustom
from PIL import Image


class WQuery(object):

    def __init__(self,dquery,media_dir,visual_index):
        self.media_dir = media_dir
        self.dquery = dquery
        self.primary_key = self.dquery.pk
        self.local_path = "{}/queries/{}.png".format(self.media_dir,self.primary_key)
        self.visual_index = visual_index

    def find(self,n):
        results = {}
        results[self.visual_index.name] = self.visual_index.nearest(image_path=self.local_path,n=n)
        return results



class WVideo(object):

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

    def extract_frames(self,args):
        frames = []
        cuts = []
        if args['rate']:
            denominator = int(args['rate'])
        else:
            denominator = 30
        if not self.dvideo.dataset:
            output_dir = "{}/{}/{}/".format(self.media_dir,self.primary_key,'frames')
            if args['rescale']:
                command = 'ffmpeg -i {} -vf "select=not(mod(n\,{})),scale={}:-1" -vsync vfr  {}/%d_b.jpg'.format(self.local_path,denominator,int(args['rescale']),output_dir)
            else:
                command = 'ffmpeg -i {} -vf "select=not(mod(n\,{}))" -vsync vfr  {}/%d_b.jpg'.format(self.local_path,denominator,output_dir)
            extract = sp.Popen(shlex.split(command))
            extract.wait()
            if extract.returncode != 0:
                raise ValueError
            for fname in glob.glob(output_dir+'*_b.jpg'):
                ind = int(fname.split('/')[-1].replace('_b.jpg', ''))
                os.rename(fname,fname.replace('{}_b.jpg'.format(ind),'{}.jpg'.format((ind-1)*denominator)))
            if not args['perform_scene_detection']:
                logging.warning("Scene detection is disabled")
            else:
                scencedetect = sp.Popen(['fab','pyscenedetect:{},{}'.format(self.primary_key,args['rescale'])],cwd=os.path.join(os.path.abspath(__file__).split('entity.py')[0],'../'))
                scencedetect.wait()
                if scencedetect.returncode != 0:
                    logging.info("pyscene detect failed with {} check fab.log for the reason".format(scencedetect.returncode))
                else:
                    with open('{}/{}/frames/scenes.json'.format(self.media_dir,self.primary_key)) as fh:
                        cuts = json.load(fh)
                    os.remove('{}/{}/frames/scenes.json'.format(self.media_dir,self.primary_key))
            frame_width, frame_height = 0, 0
            for i,fname in enumerate(glob.glob(output_dir+'*.jpg')):
                ind = int(fname.split('/')[-1].replace('.jpg', ''))
                if i == 0:
                    im = Image.open(fname)
                    frame_width, frame_height = im.size # this remains constant for all frames
                f = WFrame(frame_index=int(ind),video=self,w=frame_width,h=frame_height)
                frames.append(f)
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
                            dst = "{}/{}/frames/{}.jpg".format(self.media_dir, self.primary_key, i)
                            os.rename(fname, dst)
                            im = Image.open(dst)
                            w, h = im.size
                            f = WFrame(frame_index=i, video=self,name=fname.split('/')[-1],subdir=subdir.replace("{}/{}/frames/".format(self.media_dir, self.primary_key),''),w=w,h=h)
                            frames.append(f)
                        else:
                            logging.warning("skipping {} not a jpeg file".format(fname))
                else:
                    logging.warning("skipping {} ".format(subdir))
        return frames, cuts

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

    def index_detections(self,detections,detection_name,visual_index):
        visual_index.load()
        entries = []
        paths = []
        for i, d in enumerate(detections):
            entry = {
                'frame_index': d.frame.frame_index,
                'detection_primary_key': d.pk,
                'frame_primary_key': d.frame.pk,
                'video_primary_key': self.primary_key,
                'index': i,
                'type': 'detection'
            }
            paths.append("{}/{}/detections/{}.jpg".format(self.media_dir, self.primary_key, d.pk))
            entries.append(entry)
        features = visual_index.index_paths(paths)
        feat_fname = "{}/{}/indexes/{}_{}.npy".format(self.media_dir, self.primary_key,detection_name, visual_index.name)
        entries_fname = "{}/{}/indexes/{}_{}.json".format(self.media_dir, self.primary_key,detection_name, visual_index.name)
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

