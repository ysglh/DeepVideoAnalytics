from dvaclient import utils
from collections import defaultdict
import json, gzip


def convert_instances(fname,prefix,subset=None):
    regions_json = []
    data = defaultdict(lambda: {'image': None, 'annotations': [], 'captions': [], 'keypoints': []})
    train_data = json.load(file(fname))
    id_to_license = {k['id']: k for k in train_data['licenses']}
    id_to_category = {k['id']: k for k in train_data['categories']}
    for entry in train_data['images']:
        if subset is None or entry['id'] in subset:
            entry['license'] = id_to_license[entry['license']]
            data[entry['id']]['image'] = entry
            fname = "{}{}.jpg".format(prefix,str(entry['id']).zfill(12))
            regions_json.append(utils.create_region_json(fname,"info",0,0,0,0,entry,None,full_frame=True))
    for a in train_data['annotations']:
        if subset is None or a['image_id'] in subset:
            a['category'] = id_to_category[a['category_id']]
            x = a['bbox'][0]
            y = a['bbox'][1]
            w = a['bbox'][2]
            h = a['bbox'][3]
            object_name = 'coco_instance/{}/{}'.format(a[u'category'][u'supercategory'], a[u'category'][u'name'])
            fname = "{}{}.jpg".format(prefix,str(a['image_id']).zfill(12))
            regions_json.append(utils.create_region_json(fname,x=x,y=y,w=w,h=h,object_name=object_name,metadata=a,
                                                              text=None))
    return regions_json


def convert_captions(fname,prefix,subset=None):
    regions_json = []
    captions_train_data = json.load(file(fname))
    for annotation in captions_train_data['annotations']:
        if subset is None or annotation['image_id'] in subset:
            fname = "{}{}.jpg".format(prefix,str(annotation['image_id']).zfill(12))
            regions_json.append(utils.create_region_json(fname,"caption",0,0,0,0,None,annotation['caption'],full_frame=True))
    return regions_json


def convert_keypoints(fname,prefix,subset=None):
    regions_json = []
    keypoints_train_data = json.load(file(fname))
    kp_id_to_category = {k['id']: k for k in keypoints_train_data['categories']}
    for annotation in keypoints_train_data['annotations']:
            if subset is None or annotation['image_id'] in subset:
                annotation['category'] = kp_id_to_category[annotation['category_id']]
                x = annotation['bbox'][0]
                y = annotation['bbox'][1]
                w = annotation['bbox'][2]
                h = annotation['bbox'][3]
                fname = "{}{}.jpg".format(prefix,str(annotation['image_id']).zfill(12))
                a = annotation
                object_name = 'coco_keypoints/{}/{}'.format(a[u'category'][u'supercategory'],a[u'category'][u'name'])
                regions_json.append(utils.create_region_json(fname,x=x,y=y,w=w,h=h,object_name=object_name,
                                                             metadata=annotation,text=None))
    return regions_json

if __name__ == '__main__':
    import sys
    # subset = {int(k) for k in ['000000100500','000000108500','000000110500','000000111500','000000114500']}
    subset = None
    if sys.argv[-1] == 'train':
        with gzip.open('coco_train2017_instances_regions.gz','w') as output:
            regions = convert_instances("coco_annotations/instances_train2017.json", prefix="train2017/", subset=subset)
            output.write(json.dumps(regions))
        with gzip.open('coco_train2017_captions_regions.gz','w') as output:
            regions = convert_captions("coco_annotations/captions_train2017.json", prefix="train2017/", subset=subset)
            output.write(json.dumps(regions))
        with gzip.open('coco_train2017_keypoints_regions.gz','w') as output:
            regions = convert_keypoints("coco_annotations/person_keypoints_train2017.json", prefix="train2017/", subset=subset)
            output.write(json.dumps(regions))
    elif sys.argv[-1] == 'val':
        with gzip.open('coco_val2017_instances_regions.gz', 'w') as output:
            regions = convert_instances("coco_annotations/instances_val2017.json", prefix="val2017/", subset=subset)
            output.write(json.dumps(regions))
        with gzip.open('coco_val2017_captions_regions.gz', 'w') as output:
            regions = convert_captions("coco_annotations/captions_val2017.json", prefix="val2017/", subset=subset)
            output.write(json.dumps(regions))
        with gzip.open('coco_val2017_keypoints_regions.gz', 'w') as output:
            regions = convert_keypoints("coco_annotations/person_keypoints_val2017.json", prefix="val2017/", subset=subset)
            output.write(json.dumps(regions))
    else:
        raise ValueError("Please specify train or val")
