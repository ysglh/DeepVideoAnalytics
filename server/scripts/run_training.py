#!/usr/bin/env python
import django, os, sys, json, subprocess
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from django.conf import settings
from dvaapp.models import Region, Frame, TrainedModel, TEvent
from dvaui.view_shared import create_detector_dataset
from dvalib.yolo import trainer

if __name__ == '__main__':
    start_pk = sys.argv[-1]
    start = TEvent.objects.get(pk=start_pk)
    args = start.arguments
    labels = set(args['labels']) if 'labels' in args else set()
    object_names = set(args['object_names']) if 'object_names' in args else set()
    detector = TrainedModel.objects.get(pk=args['detector_pk'])
    detector.create_directory()
    args['root_dir'] = "{}/detectors/{}/".format(settings.MEDIA_ROOT, detector.pk)
    args['base_model'] = "{}/detectors/yolo/yolo.h5"
    class_distribution, class_names, rboxes, rboxes_set, frames, i_class_names = create_detector_dataset(object_names,
                                                                                                         labels)
    images, boxes = [], []
    path_to_f = {}
    for k, f in frames.iteritems():
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT, f.video_id, f.frame_index)
        path_to_f[path] = f
        images.append(path)
        boxes.append(rboxes[k])
        # print k,rboxes[k]
    with open("{}/input.json".format(args['root_dir']), 'w') as input_data:
        json.dump({'boxes': boxes,
                   'images': images,
                   'args': args,
                   'class_names': class_names.items(),
                   'class_distribution': class_distribution.items()},
                  input_data)
    detector.boxes_count = sum([len(k) for k in boxes])
    detector.frames_count = len(images)
    detector.classes_count = len(class_names)
    detector.save()
    args['class_names'] = i_class_names
    train_task = trainer.YOLOTrainer(boxes=boxes, images=images, args=args)
    train_task.train()
    detector.phase_1_log = file("{}/phase_1.log".format(args['root_dir'])).read()
    detector.phase_2_log = file("{}/phase_2.log".format(args['root_dir'])).read()
    detector.class_distribution = json.dumps(class_distribution.items())
    detector.class_names = json.dumps(class_names.items())
    detector.trained = True
    detector.save()
    results = train_task.predict()
    bulk_regions = []
    for path, box_class, score, top, left, bottom, right in results:
        r = Region()
        r.region_type = r.ANNOTATION
        r.confidence = int(100.0 * score)
        r.object_name = "YOLO_{}_{}".format(detector.pk, box_class)
        r.y = top
        r.x = left
        r.w = right - left
        r.h = bottom - top
        r.frame_id = path_to_f[path].pk
        r.video_id = path_to_f[path].video_id
        bulk_regions.append(r)
    Region.objects.bulk_create(bulk_regions, batch_size=1000)
    folder_name = "{}/detectors/{}".format(settings.MEDIA_ROOT, detector.pk)
    file_name = '{}/exports/{}.dva_detector.zip'.format(settings.MEDIA_ROOT, detector.pk)
    zipper = subprocess.Popen(['zip', file_name, '-r', '.'], cwd=folder_name)
    zipper.wait()
