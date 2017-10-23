import os,logging,time,boto3, glob,subprocess,calendar,sys
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
import json
import random
import gzip
import shutil
from urllib import urlretrieve
from collections import defaultdict
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M',filename='fab.log',filemode='a')


@task
def kill():
    try:
        local("ps auxww | grep 'celery -A dva worker * ' | awk '{print $2}' | xargs kill -9")
    except:
        pass


def setup_django():
    import django
    sys.path.append("../../")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()


def get_coco_dirname():
    if sys.platform == 'darwin':
        dirname = '/Users/aub3/coco_input/'
    else:
        dirname = 'coco_input'
    return dirname


@task
def create_yolo_test_data():
    import shutil
    import numpy as np
    import os
    from PIL import Image
    setup_django()
    from dvaapp.view_shared import handle_uploaded_file
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.models import Region,TEvent,Frame,Label,RegionLabel
    from dvaapp.tasks import perform_dataset_extraction,perform_export
    try:
        shutil.rmtree('/Users/aub3/tests/yolo_test')
    except:
        pass
    try:
        os.mkdir('/Users/aub3/tests/yolo_test')
    except:
        pass
    data = np.load('shared/underwater_data.npz')
    json_test = {}
    json_test['anchors'] = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]
    id_2_boxes = {}
    class_names = {
        0:"red_buoy",
        1:"green_buoy",
        2:"yellow_buoy",
        3:"path_marker",
        4:"start_gate",
        5:"channel"
    }
    labels = {k: Label.objects.create(name=v, set="test") for k, v in class_names}
    for i,image in enumerate(data['images'][:500]):
        path = "/Users/aub3/tests/yolo_test/{}.jpg".format(i)
        Image.fromarray(image).save(path)
        id_2_boxes[path.split('/')[-1]] = data['boxes'][i].tolist()
    local('zip /Users/aub3/tests/yolo_test.zip -r /Users/aub3/tests/yolo_test/* ')
    fname = "/Users/aub3/tests/yolo_test.zip"
    name = "yolo_test"
    f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
    dv = handle_uploaded_file(f, name)
    perform_dataset_extraction(TEvent.objects.create(video=dv).pk)
    for df in Frame.objects.filter(video=dv):
        for box in id_2_boxes[df.name]:
            r = Region()
            r.video = dv
            r.frame = df
            c , top_x, top_y, bottom_x, bottom_y = box
            r.object_name = class_names[c]
            r.region_type = Region.ANNOTATION
            r.x = top_x
            r.y = top_y
            r.w = bottom_x - top_x
            r.h = bottom_y - top_y
            r.save()
            l = RegionLabel()
            l.frame = df
            l.video = dv
            l.label = labels[c]
            l.region = r
            l.save()
    perform_export(TEvent.objects.create(video=dv,arguments={'destination':'FILE'}).pk)
    try:
        shutil.rmtree('/Users/aub3/tests/yolo_test')
    except:
        pass


def process_visual_genome():
    setup_django()
    import os, shutil, gzip, json
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.view_shared import handle_uploaded_file
    from dvaapp import models
    from dvaapp.models import TEvent
    from dvaapp.tasks import perform_dataset_extraction, export_video
    from collections import defaultdict
    os.system('aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key visual_genome_objects.txt.gz  /root/DVA/visual_genome_objects.txt.gz')
    data = defaultdict(list)
    with gzip.open('/root/DVA/visual_genome_objects.txt.gz') as metadata:
        for line in metadata:
            entries = line.strip().split('\t')
            data[entries[1]].append({
                'x': int(entries[2]),
                'y': int(entries[3]),
                'w': int(entries[4]),
                'h': int(entries[5]),
                'object_id': entries[0],
                'object_name': entries[6],
                'text': ' '.join(entries[6:]), })
    name = "visual_genome"
    fname = "visual_genome.zip"
    f = SimpleUploadedFile(fname, "", content_type="application/zip")
    v = handle_uploaded_file(f, name)
    outpath = "/root/DVA/dva/media/{}/video/{}.zip".format(v.pk, v.pk)
    os.system('rm  {}'.format(outpath))
    os.system(
        'aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key visual_genome.zip  {}'.format(
            outpath))
    perform_dataset_extraction(TEvent.objects.create(video=v).pk)
    video = v
    models.Region.objects.all().filter(video=video).delete()
    buffer = []
    batch_count = 0
    for frame in models.Frame.objects.all().filter(video=video):
        frame_id = str(int(frame.name.split('/')[-1].split('.')[0]))
        for o in data[frame_id]:
            annotation = models.Region()
            annotation.region_type = models.Region.ANNOTATION
            annotation.video = v
            annotation.frame = frame
            annotation.x = o['x']
            annotation.y = o['y']
            annotation.h = o['h']
            annotation.w = o['w']
            annotation.object_name = o['object_name']
            annotation.metadata = o
            annotation.text = o['text']
            buffer.append(annotation)
            if len(buffer) == 1000:
                try:
                    models.Region.objects.bulk_create(buffer)
                    batch_count += 1
                    print "saved {}".format(batch_count)
                except:
                    print "encountered an error doing one by one"
                    for k in buffer:
                        try:
                            k.save()
                        except:
                            print "skipping"
                            print k.object_name
                buffer = []
    try:
        models.Region.objects.bulk_create(buffer)
        print "saved {}".format(batch_count)
    except:
        print "encountered an error doing one by one"
        for k in buffer:
            try:
                k.save()
            except:
                print "skipping"
                print k.object_name
    print "exporting"
    export_video(TEvent.objects.create(video=v).pk)


@task
def create_visual_genome():
    """
    Create Visual Genome dataset
    :return:
    """
    kill()
    prompt = "Note running this outside US-East-1 EC2 region will cost 1.5$ due to bandwidth consumed type 'yes' to proceed "
    if raw_input(prompt) == 'yes':
        process_visual_genome()


@task
def download_coco(size=500):
    dirname = get_coco_dirname()
    try:
        os.mkdir(dirname)
        with lcd(dirname):
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip")
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip")
            local("wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip")
            local('unzip "*.zip"')
    except:
        pass
    train_data = json.load(file("{}/annotations/instances_train2014.json".format(dirname)))
    captions_train_data = json.load(file("{}/annotations/captions_train2014.json".format(dirname)))
    keypoints_train_data = json.load(file("{}/annotations/person_keypoints_train2014.json".format(dirname)))
    sample = random.sample(train_data['images'], int(size))
    ids = set()
    for count, img in enumerate(sample):
        if (count + 1) % 100 == 0:
            print count
        fname = os.path.join(dirname, img['file_name'])
        if not os.path.exists(fname):
            urlretrieve(img['coco_url'], fname)
        ids.add(img['id'])
    data = defaultdict(lambda: {'image': None, 'annotations': [], 'captions': [], 'keypoints': []})
    id_to_license = {k['id']: k for k in train_data['licenses']}
    id_to_category = {k['id']: k for k in train_data['categories']}
    kp_id_to_category = {k['id']: k for k in keypoints_train_data['categories']}
    for entry in train_data['images']:
        if entry['id'] in ids:
            entry['license'] = id_to_license[entry['license']]
            data[entry['id']]['image'] = entry
    for annotation in train_data['annotations']:
        if annotation['image_id'] in ids:
            annotation['category'] = id_to_category[annotation['category_id']]
            data[annotation['image_id']]['annotations'].append(annotation)
    del train_data
    for annotation in captions_train_data['annotations']:
        if annotation['image_id'] in ids:
            data[annotation['image_id']]['captions'].append(annotation)
    del captions_train_data
    for annotation in keypoints_train_data['annotations']:
        if annotation['image_id'] in ids:
            annotation['category'] = kp_id_to_category[annotation['category_id']]
            data[annotation['image_id']]['keypoints'].append(annotation)
    del keypoints_train_data
    with open('{}/coco_sample_metadata.json'.format(dirname), 'w') as output:
        json.dump(data, output)


def create_coco_annotations_file():
    """
    #!/usr/bin/env bash
    aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/COCO_Text.zip  COCO_Text.zip
    aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/captions.zip  captions.zip
    aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/instances.zip  instances.zip
    aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/persons.zip  persons.zip
    unzip COCO_Text.zip
    unzip captions.zip
    unzip instances.zip
    unzip persons.zip

    import sys,os,random
    import django
    sys.path.append("../../")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    import os, shutil, gzip, json
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.view_shared import handle_uploaded_file
    from dvaapp import models
    from dvaapp.models import TEvent
    from dvaapp.tasks import perform_dataset_extraction,perform_export
    from collections import defaultdict


    def create_annotations():
        train_data = json.load(file('annotations/instances_train2014.json'))
        captions_train_data = json.load(file('annotations/captions_train2014.json'))
        keypoints_train_data = json.load(file('annotations/person_keypoints_train2014.json'))
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
        return data


    def create_dataset():
        name = "coco_train"
        fname = "coco_train.zip"
        f = SimpleUploadedFile(fname, "", content_type="application/zip")
        v = handle_uploaded_file(f, name)
        outpath = "/root/DVA/dva/media/{}/video/{}.zip".format(v.pk, v.pk)
        os.system('rm  {}'.format(outpath))
        command = 'aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key coco/train2014.zip  {}'.format(outpath)
        print command
        os.system(command)
        perform_dataset_extraction(TEvent.objects.create(video=v).pk)


    def load_annotations(v,data):
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
                annotation.object_name = 'coco_instance/{}/{}'.format(a[u'category'][u'supercategory'],
                                                                      a[u'category'][u'name'])
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
                annotation.object_name = 'coco_keypoints/{}/{}'.format(a[u'category'][u'supercategory'],
                                                                       a[u'category'][u'name'])
                buf.append(annotation)
            for a in data[frame_id][u'text']:
                annotation = models.Region()
                annotation.region_type = models.Region.ANNOTATION
                annotation.video_id = v.pk
                annotation.frame_id = frame.pk
                annotation.text = a['utf8_string'] if 'utf8_string' in a else ""
                annotation.metadata = a
                annotation.x = a['bbox'][0]
                annotation.y = a['bbox'][1]
                annotation.w = a['bbox'][2]
                annotation.h = a['bbox'][3]
                annotation.object_name = 'text/{}/{}/{}'.format(a['class'], a['legibility'], a['language'])
                buf.append(annotation)
            for caption in data[frame_id][u'captions']:
                annotation = models.Region()
                annotation.region_type = models.Region.ANNOTATION
                annotation.video_id = v.pk
                annotation.frame_id = frame.pk
                annotation.text = caption['caption']
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

    def export():
        pass

    if __name__ == '__main__':
        pass
    :return:
    """
    pass