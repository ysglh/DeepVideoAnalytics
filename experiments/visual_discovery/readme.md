
<center><h1>Building a Visual Discovery app using Deep Video Analytics</h1></center>

<center><h2>Backend & Database</h2></center>

### Setup & import Django app


```python
import sys
sys.path.append("../../")
import django,os,glob,base64
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
django.setup()
from IPython.display import display,Image
```


```python
import dvaapp.views as views
import dvaapp.tasks as tasks
from dvaapp.models import TEvent
from django.conf import settings
```

## Extraction and processing videos
You can monitor progress at https://localhost:8000/
Note that the following code executes each task in a synchronous manner, however when you submit videos through 
the web UI the tasks are executed in async manner using background celery workers. Further upon successful execution of each task the next task in processing pipeline is automatically created. 


```python
video_list = [('vogue fashion',"https://www.youtube.com/watch?v=TJA6I-ZaoTY"),
              ('home decoration','https://www.youtube.com/watch?v=f0BkV6OV8z4')]
videos = {}
for name,url in video_list:
    # Create a video object
    v = views.handle_youtube_video(name,url)
    # Extract frames
    tasks.extract_frames(TEvent.objects.create(video=v).pk)
    # Perform inception
    tasks.inception_index_by_id(TEvent.objects.create(video=v).pk)
    # Perform SSD Detection
    tasks.perform_ssd_detection_by_id(TEvent.objects.create(video=v).pk)
    # Perform indexing on faces detected in the video
    tasks.perform_face_indexing(v.pk)
    # Index bounding boxeson all region
    tasks.inception_index_regions_by_id(TEvent.objects.create(video=v).pk)
```

    WARNING:root:Loading the network inception , first apply / query will be slower


## A simple query


```python
Image(filename='query_chair.png')
```




![png](output_8_0.png)



## Results


```python
query,dv = views.create_query(20,False,['inception',],[],'data:image/png;base64,'+base64.encodestring(file('query_chair.png').read()))
results = tasks.inception_query_by_image(query.pk)
```

### Top 20 results


```python
for k in results['inception'][:20]:
    if k[u'type'] == 'D' or k[u'type'] == 'A':
        print "Detection"
        display(Image(filename="{}/{}/detections/{}.jpg".format(settings.MEDIA_ROOT,k[u'video_primary_key'],k[u'detection_primary_key'])))
    else:
        print "Frame"
        display(Image(filename="{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT,k[u'video_primary_key'],k[u'frame_index'])))
```

    Detection



![jpeg](output_12_1.jpeg)


    Detection



![jpeg](output_12_3.jpeg)


    Detection



![jpeg](output_12_5.jpeg)


    Detection



![jpeg](output_12_7.jpeg)


    Detection



![jpeg](output_12_9.jpeg)


    Detection



![jpeg](output_12_11.jpeg)


    Frame



![jpeg](output_12_13.jpeg)


    Detection



![jpeg](output_12_15.jpeg)


    Detection



![jpeg](output_12_17.jpeg)


    Detection



![jpeg](output_12_19.jpeg)


    Frame



![jpeg](output_12_21.jpeg)


    Detection



![jpeg](output_12_23.jpeg)


    Detection



![jpeg](output_12_25.jpeg)


    Detection



![jpeg](output_12_27.jpeg)


    Frame



![jpeg](output_12_29.jpeg)


    Frame



![jpeg](output_12_31.jpeg)


    Detection



![jpeg](output_12_33.jpeg)


    Detection



![jpeg](output_12_35.jpeg)


    Frame



![jpeg](output_12_37.jpeg)


    Detection



![jpeg](output_12_39.jpeg)



```python

```
