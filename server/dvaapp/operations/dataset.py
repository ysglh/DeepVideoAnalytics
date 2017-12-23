import json,os,zipfile,logging
from PIL import Image
from ..models import Frame, Region, Label, FrameLabel
from collections import defaultdict


class DatasetCreator(object):
    """
    Wrapper object for a  dataset
    """

    def __init__(self,dvideo,media_dir):
        self.dvideo = dvideo
        self.primary_key = self.dvideo.pk
        self.media_dir = media_dir
        self.local_path = dvideo.path()
        self.segments_dir = "{}/{}/segments/".format(self.media_dir,self.primary_key)
        self.duration = None
        self.width = None
        self.height = None
        self.metadata = {}
        self.segment_frames_dict = {}
        self.csv_format = None

    def extract(self,extract_event):
        self.extract_zip_dataset(extract_event)
        os.remove("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key))

    def extract_zip_dataset(self,event):
        zipf = zipfile.ZipFile("{}/{}/video/{}.zip".format(self.media_dir, self.primary_key, self.primary_key), 'r')
        zipf.extractall("{}/{}/frames/".format(self.media_dir, self.primary_key))
        zipf.close()
        i = 0
        df_list = []
        root_length = len("{}/{}/frames/".format(self.media_dir, self.primary_key))
        for subdir, dirs, files in os.walk("{}/{}/frames/".format(self.media_dir, self.primary_key)):
            if '__MACOSX' not in subdir:
                for ofname in files:
                    fname = os.path.join(subdir, ofname)
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
                            df.event_id = event.pk
                            df.name = os.path.join(subdir[root_length:], ofname)
                            if not df.name.startswith('/'):
                                df.name = "/{}".format(df.name)
                            s = "/{}/".format(subdir[root_length:]).replace('//','/')
                            df.subdir = s
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
                for l in f.subdir.split('/')[1:]:
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
