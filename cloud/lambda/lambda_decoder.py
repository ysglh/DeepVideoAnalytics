import time
import shlex,json,os,zipfile,logging
import subprocess as sp
try:
    from dvalib import indexer, clustering, retriever
    import numpy as np
except ImportError:
    np = None
    logging.warning("Could not import indexer / clustering assuming running in front-end mode / Heroku")
from collections import defaultdict

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

def parse_segment_framelist(csv_format, segment_id, framelist):
    if csv_format is None:
        detect_csv_segment_format()
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


def decode_segment(self,ds,denominator,rescale):
    output_dir = "{}/{}/{}/".format(self.media_dir, self.primary_key, 'frames')
    input_segment = ds.path()
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
    with open(ds.framelist_path()) as framelist:
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
    #     if i ==0:
    #         im = Image.open(dst)
    #         frame_width, frame_height = im.size  # this remains constant for all frames
    #     df = Frame()
    #     df.frame_index = int(frame_index+ds.start_index)
    #     df.video_id = self.dvideo.pk
    #     df.keyframe = True if frame_data['type'] == 'I' else False
    #     df.t = frame_data['ts']
    #     df.segment_index = ds.segment_index
    #     df.h = frame_height
    #     df.w = frame_width
    #     df_list.append(df)
    # _ = Frame.objects.bulk_create(df_list, batch_size=1000)

