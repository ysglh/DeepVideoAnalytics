import sys,os,random
import django
sys.path.append("../../")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
import os, shutil, gzip, json
from django.core.files.uploadedfile import SimpleUploadedFile
from dvaapp.shared import handle_uploaded_file
from dvaapp import models
from dvaapp.models import TEvent
from dvaapp.tasks import extract_frames, export_video
from collections import defaultdict




os.system('aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/COCO_Text.zip  COCO_Text.zip')
os.system('aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/captions.zip  captions.zip')
os.system('aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/instances.zip  instances.zip')
os.system('aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/persons.zip  persons.zip')

os.system('unzip COCO_Text.zip')
os.system('unzip captions.zip')
os.system('unzip instances.zip')
os.system('unzip persons.zip')

train_data = json.load(file('annotations/instances_val2014.json'))
captions_train_data = json.load(file('annotations/captions_val2014.json'))
keypoints_train_data = json.load(file('annotations/instances_val2014.json'))
text_train_data = json.load(file('COCO_Text.json'))
data = defaultdict(lambda: {'image': None, 'annotations': [], 'captions': [], 'keypoints': [], 'text': []})
id_to_license = {k['id']: k for k in train_data['licenses']}
id_to_category = {k['id']: k for k in train_data['categories']}
kp_id_to_category = {k['id']: k for k in keypoints_train_data['categories']}
for entry in train_data['images']:
    entry['license'] = id_to_license[entry['license']]
    data[entry['id']]['image'] = entry
for annotation in train_data['annotations']:
    annotation['category'] = id_to_category[annotation['category_id']]
    data[annotation['image_id']]['annotations'].append(annotation)
del train_data
for annotation in captions_train_data['annotations']:
    data[annotation['image_id']]['captions'].append(annotation)
del captions_train_data
for annotation in keypoints_train_data['annotations']:
    annotation['category'] = kp_id_to_category[annotation['category_id']]
    data[annotation['image_id']]['keypoints'].append(annotation)
for annotation in text_train_data['anns'].itervalues():
    data[annotation['image_id']]['text'].append(annotation)
name = "coco_val"
fname = "coco_val.zip"
f = SimpleUploadedFile(fname, "", content_type="application/zip")
v = handle_uploaded_file(f, name)
outpath = "/root/DVA/dva/media/{}/video/{}.zip".format(v.pk, v.pk)
os.system('rm  {}'.format(outpath))
command = 'aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/val2014.zip  {}'.format(
        outpath)
print command
os.system(command)
extract_frames(TEvent.objects.create(video=v).pk)
video = v
buf = []
batch_count = 0
models.Region.objects.all().filter(video=v).delete()
for frame in models.Frame.objects.all().filter(video=video):
    frame_id = int(frame.name.split('_')[-1].split('.')[0])
    annotation = models.Region()
    annotation.region_type = models.Region.ANNOTATION
    annotation.video_id = v.pk
    annotation.frame_id = frame.pk
    annotation.full_frame = True
    annotation.metadata = data[frame_id]['image']
    annotation.object_name = 'metadata'
    buf.append(annotation)
    for a in data[frame_id][u'annotations']:
        annotation = models.Region()
        annotation.region_type = models.Region.ANNOTATION
        annotation.video_id = v.pk
        annotation.frame_id = frame.pk
        annotation.metadata = a
        annotation.full_frame = False
        annotation.x = a['bbox'][0]
        annotation.y = a['bbox'][1]
        annotation.w = a['bbox'][2]
        annotation.h = a['bbox'][3]
        annotation.object_name = 'coco_instance/{}/{}'.format(a[u'category'][u'supercategory'], a[u'category'][u'name'])
        buf.append(annotation)
    for a in data[frame_id][u'keypoints']:
        annotation = models.Region()
        annotation.region_type = models.Region.ANNOTATION
        annotation.video_id = v.pk
        annotation.frame_id = frame.pk
        annotation.metadata = a
        annotation.x = a['bbox'][0]
        annotation.y = a['bbox'][1]
        annotation.w = a['bbox'][2]
        annotation.h = a['bbox'][3]
        annotation.object_name = 'coco_keypoints/{}/{}'.format(a[u'category'][u'supercategory'], a[u'category'][u'name'])
        buf.append(annotation)
    for a in data[frame_id][u'text']:
        annotation = models.Region()
        annotation.region_type = models.Region.ANNOTATION
        annotation.video_id = v.pk
        annotation.frame_id = frame.pk
        annotation.metadata_text = a['utf8_string'] if 'utf8_string' in a else ""
        annotation.metadata = a
        annotation.x = a['bbox'][0]
        annotation.y = a['bbox'][1]
        annotation.w = a['bbox'][2]
        annotation.h = a['bbox'][3]
        annotation.object_name = 'text/{}/{}/{}'.format(a['class'],a['legibility'],a['language'])
        buf.append(annotation)
    for caption in data[frame_id][u'captions']:
        annotation = models.Region()
        annotation.region_type = models.Region.ANNOTATION
        annotation.video_id = v.pk
        annotation.frame_id = frame.pk
        annotation.metadata_text = caption['caption']
        annotation.full_frame = True
        annotation.object_name = 'caption'
        buf.append(annotation)
    if len(buf) > 1000:
        try:
            models.Region.objects.bulk_create(buf)
            batch_count += 1
            print "saved {}".format(batch_count)
        except:
            print "encountered an error doing one by one"
            for k in buf:
                try:
                    k.save()
                except:
                    print "skipping"
                    print k.object_name
        buf = []
try:
    models.Region.objects.bulk_create(buf)
    batch_count += 1
    print "saved {}".format(batch_count)
except:
    print "encountered an error doing one by one"
    for k in buf:
        try:
            k.save()
        except:
            print "skipping"
            print k.object_name
buf = []
export_video(TEvent.objects.create(video=v).pk)