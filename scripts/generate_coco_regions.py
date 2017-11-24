import json
from collections import defaultdict
import data_utils


def convert_instances(fname,test=False):
    regions_json = []
    data = defaultdict(lambda: {'image': None, 'annotations': [], 'captions': [], 'keypoints': []})
    train_data = json.load(file(fname))
    id_to_license = {k['id']: k for k in train_data['licenses']}
    id_to_category = {k['id']: k for k in train_data['categories']}
    for entry in train_data['images']:
        entry['license'] = id_to_license[entry['license']]
        data[entry['id']]['image'] = entry
        fname = "{}.jpg".format(entry['image_id'])


        regions_json.append(data_utils.create_region_json(fname,0,0,0,0,None,entry,full_frame=True))

    for a in train_data['annotations']:
        a['category'] = id_to_category[a['category_id']]
        x = a['bbox'][0]
        y = a['bbox'][1]
        w = a['bbox'][2]
        h = a['bbox'][3]
        object_name = 'coco_instance/{}/{}'.format(a[u'category'][u'supercategory'], a[u'category'][u'name'])
        fname = "{}.jpg".format(a['image_id'])
        regions_json.append(data_utils.create_region_json(fname,x=x,y=y,w=w,h=h,object_name=object_name,metadata=a,
                                                          text=None))
    return regions_json


def convert_captions(fname,test=False):
    regions_json = []
    captions_train_data = json.load(file(fname))
    for annotation in captions_train_data['annotations']:
        fname = "{}.jpg".format(annotation['image_id'])
        regions_json.append(data_utils.create_region_json(fname,0,0,0,0,None,annotation,full_frame=True))
    return regions_json


def convert_keypoints(fname,test=False):
    regions_json = []
    keypoints_train_data = json.load(file(fname))
    kp_id_to_category = {k['id']: k for k in keypoints_train_data['categories']}
    for annotation in keypoints_train_data['annotations']:
            annotation['category'] = kp_id_to_category[annotation['category_id']]
            fname = "{}.jpg".format(annotation['image_id'])
    return regions_json

if __name__ == '__main__':
    regions = []
    regions += convert_instances("coco_annotations/instances_train2017.json")
    regions += convert_captions("coco_annotations/captions_train2017.json")
    regions += convert_keypoints("coco_annotations/person_keypoints_train2014.json")
    with open('coco_annotations/coco_train_dva_import.json','w') as output:
        output.write(json.dumps(regions))
