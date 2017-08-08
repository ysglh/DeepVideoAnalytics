import time
import shlex,json,os,zipfile,logging
import subprocess as sp
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")

from PIL import Image


from ..models import Frame, Segment, Label, FrameLabel
from collections import defaultdict


class VideoDecoder(object):
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

    def extract(self,args,start):
        self.extract_zip_dataset(start)

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

    def extract_zip_dataset(self,event):
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
                        labels_to_frame[l].add((df_ids[i].id,f.frame_index))
        label_list = []
        for l in labels_to_frame:
            dl, _ = Label.objects.get_or_create(name=l,set="Directory")
            for fpk,frame_index in labels_to_frame[l]:
                a = FrameLabel()
                a.video_id = self.dvideo.pk
                a.frame_id = fpk
                a.frame_index = frame_index
                a.label_id = dl.pk
                a.event_id = event.pk
                label_list.append(a)
        FrameLabel.objects.bulk_create(label_list, batch_size=1000)
