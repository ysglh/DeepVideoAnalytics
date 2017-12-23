import os, logging, sys
from fabric.api import task, local

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', filename='fab.log', filemode='a')


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


@task
def create_yolo_test_data():
    import shutil
    import numpy as np
    import os
    from PIL import Image
    setup_django()
    from dvaui.view_shared import handle_uploaded_file
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaapp.models import Region, TEvent, Frame, Label, RegionLabel
    from dvaapp.tasks import perform_dataset_extraction, perform_export
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
    json_test['anchors'] = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778),
                            (9.77052, 9.16828)]
    id_2_boxes = {}
    class_names = {
        0: "red_buoy",
        1: "green_buoy",
        2: "yellow_buoy",
        3: "path_marker",
        4: "start_gate",
        5: "channel"
    }
    labels = {k: Label.objects.create(name=v, set="test") for k, v in class_names}
    for i, image in enumerate(data['images'][:500]):
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
            c, top_x, top_y, bottom_x, bottom_y = box
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
    perform_export(TEvent.objects.create(video=dv, arguments={'destination': 'FILE'}).pk)
    try:
        shutil.rmtree('/Users/aub3/tests/yolo_test')
    except:
        pass


def process_visual_genome():
    setup_django()
    import os, gzip
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file
    from dvaapp import models
    from dvaapp.models import TEvent
    from dvaapp.tasks import perform_dataset_extraction, perform_export
    from collections import defaultdict
    os.system(
        'aws s3api get-object --request-payer "requester" --bucket visualdatanetwork --key visual_genome_objects.txt.gz  /root/DVA/visual_genome_objects.txt.gz')
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
    perform_export(TEvent.objects.create(video=v).pk)


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
