import json
import random
import os
import shutil
from urllib import urlretrieve
import sys
sys.path.append('../../')
import django,os,glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
os.system("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip")
os.system("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip")
os.system("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip")
os.system("unzip *")
train_data = json.load(file("annotations/instances_train2014.json"))
capitons_train_data = json.load(file("annotations/captions_train2014.json"))
keypoints_train_data = json.load(file("annotations/person_keypoints_train2014.json"))
sample = random.sample(train_data['images'],500)
dirname = 'coco_input/'
ids = set()
try:
    shutil.rmtree(dirname)
except:
    pass
os.mkdir(dirname)
for count,img in enumerate(sample):
    if (count+1)%100== 0:
        print count
    fname = os.path.join(dirname, img['file_name'])
    if not os.path.exists(fname):
        urlretrieve(img['coco_url'], fname)
    ids.add(img['id'])
from collections import defaultdict
data = defaultdict(lambda:{'image':None,'annotations':[],'captions':[],'keypoints':[]})
id_to_license = {k['id']:k for k in train_data['licenses']}
id_to_category = {k['id']:k for k in train_data['categories']}
kp_id_to_category = {k['id']:k for k in keypoints_train_data['categories']}
for entry in train_data['images']:
    if entry['id'] in ids:
        entry['license'] = id_to_license[entry['license']]
        data[entry['id']]['image'] = entry
for annotation in train_data['annotations']:
    if annotation['image_id'] in ids:
        annotation['category']=id_to_category[annotation['category_id']]
        data[annotation['image_id']]['annotations'].append(annotation)
for annotation in capitons_train_data['annotations']:
    if annotation['image_id'] in ids:
        data[annotation['image_id']]['captions'].append(annotation)
for annotation in keypoints_train_data['annotations']:
    if annotation['image_id'] in ids:
        annotation['category']=kp_id_to_category[annotation['category_id']]
        data[annotation['image_id']]['keypoints'].append(annotation)
with open('coco_sample_metadata.json','w') as output:
    json.dump(data,output)
os.system("zip coco_input.zip -r {}".format(dirname))
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaapp.views import handle_uploaded_file, handle_youtube_video
from dvaapp.models import Video
from dvaapp import models
from dvaapp.tasks import extract_frames, perform_face_indexing, inception_index_by_id, perform_ssd_detection_by_id, perform_yolo_detection_by_id, inception_index_ssd_detection_by_id
fname = 'coco_input.zip'
f = SimpleUploadedFile(fname, file('coco_input.zip').read(), content_type="application/zip")
v = handle_uploaded_file(f, 'mscoco_sample_500')
extract_frames(v.pk)
data = json.load(file('coco_sample_metadata.json'))
video=v
models.Annotation.objects.all().filter(video=video).delete()
for frame in models.Frame.objects.all().filter(video=video):
    frame_id = str(int(frame.name.split('_')[-1].split('.')[0]))
    annotation = models.Annotation()
    annotation.video = v
    annotation.frame = frame
    annotation.full_frame = True
    annotation.metadata_text = json.dumps(data[frame_id]['image'])
    annotation.label = 'metadata'
    annotation.save()
for frame in models.Frame.objects.all().filter(video=video):
    frame_id = str(int(frame.name.split('_')[-1].split('.')[0]))
    for a in data[frame_id][u'annotations']:
        annotation = models.Annotation()
        annotation.video = v
        annotation.frame = frame
        annotation.metadata_text = json.dumps(a)
        annotation.full_frame = False
        annotation.x = a['bbox'][0]
        annotation.y = a['bbox'][1]
        annotation.w = a['bbox'][2]
        annotation.h = a['bbox'][3]
        label,_ = models.VLabel.objects.get_or_create(video=video,label_name='coco_instance/{}/{}'.format(a[u'category'][u'supercategory'],a[u'category'][u'name']))
        annotation.label = label.label_name
        annotation.label_parent = label
        annotation.save()
    for a in data[frame_id][u'keypoints']:
        annotation = models.Annotation()
        annotation.video = v
        annotation.frame = frame
        annotation.metadata_text = json.dumps(a)
        annotation.full_frame = False
        annotation.x = a['bbox'][0]
        annotation.y = a['bbox'][1]
        annotation.w = a['bbox'][2]
        annotation.h = a['bbox'][3]
        label,_ = models.VLabel.objects.get_or_create(video=video,label_name='coco_keypoints/{}/{}'.format(a[u'category'][u'supercategory'],a[u'category'][u'name']))
        annotation.label = label.label_name
        annotation.label_parent = label
        annotation.save()
    for caption in data[frame_id][u'captions']:
        annotation = models.Annotation()
        annotation.video = v
        annotation.frame = frame
        annotation.metadata_text = json.dumps(caption)
        annotation.full_frame = True
        label,_ = models.VLabel.objects.get_or_create(video=video,label_name='coco_caption')
        annotation.label = label.label_name
        annotation.label_parent = label
        annotation.save()
from dvaapp.tasks import export_video_by_id
from django.conf import settings
# inception_index_by_id(v.pk)
# perform_ssd_detection_by_id(v.pk)
# perform_yolo_detection_by_id(v.pk)
# perform_face_indexing(v.pk)
# inception_index_ssd_detection_by_id(v.pk)
export_video_by_id(v.pk)
os.system('cp {}/exports/{}*.dva_export.zip .'.format(settings.MEDIA_ROOT,v.pk))
