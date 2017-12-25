import django
sys.path.append("../../")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
import shutil
import numpy as np
import os
from PIL import Image
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

