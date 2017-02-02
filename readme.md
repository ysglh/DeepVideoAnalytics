#Deep Video Analytics
#### (Under development, please come back for updates)

### Installation using docker-compose

````bash
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics 
cd DeepVideoAnalytics/docker
docker-compose up 
````

### Distributed architecture

- Metadata stored in Postgres.
- Operations (Querying, Frame extraction & Indexing) performed using celery tasks and RabbitMQ.
- Separate queues and workers to allow select machines with GPU & RAM for indexing / computing features.
- Videos, frames, indexes, numpy vectors stored in media directory.

### Simple schema for extensibility
```
media/
├── 1
│   ├── audio
│   ├── detections
│   ├── frames
│   │   ├── 0.jpg
│   │   ├── 10.jpg
│   │   ...
│   │   └── 98.jpg
│   ├── indexes
│   │   ├── alexnet.framelist
│   │   └── alexnet.npy
│   └── video
│       └── 1.mp4
├── 2
│   └── video
│       └── 2.mp4
│   ├── detections
│   ├── frames
│   │   ├── 0.jpg
│   │   ├── 10.jpg
...
└── queries
    ├── 1.png
    ├── 10.png
    ....
    ├── 8.png
    └── 9.png

19 directories, 257 files
```

# Libraries & Code used
- Pytorch (specifically alexnet.py) [License](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- Darknet (http://pjreddie.com/darknet/) [License](https://github.com/pjreddie/darknet/blob/master/LICENSE)
- AdminLTE2 (https://almsaeedstudio.com/) [License](https://github.com/almasaeed2010/AdminLTE/blob/master/LICENSE)

# Copyright
Copyright, Akshay Bhat, Cornell University, All rights reserved.