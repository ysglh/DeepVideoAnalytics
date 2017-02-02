#Deep Video Analytics
**(Under active development, please come back for updates)**

### Installation using docker-compose

````bash
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics 
cd DeepVideoAnalytics/docker
docker-compose up 
````
### User Interface (Alpha version still under development)

![UI Screenshot](demo/alpha_screenshot.png "Alpha Screenshot")



### Implemented algorithms
 (GPU version not yet implemented/tested)
 
 - Alexnet indexing using Pytorch 
 - Darkenet YOLO 9000 detections
 - Google inception using Tensorflow
   **(Ongoing. Porting code from Visual Search Server)**
 
### Distributed architecture

![Architecture](demo/architecture.png "System architecture")

- Metadata stored in Postgres.
- Operations (Querying, Frame extraction & Indexing) performed using celery tasks and RabbitMQ.
- Separate queues and workers to allow select machines with GPU & RAM for indexing / computing features.
- Videos, frames, indexes, numpy vectors stored in media directory.

### Explore without User Interface

You can use the jupyter notebook explore.ipynb to manually run tasks & code against the databases. 

### Simple schema for extensibility

 - One directory per video or dataset (a set of images)
 - Extracted frames and detections are stored in detections/ & frames/ under the video directory
 - Indexes (numpy arrays) and list of corresponding frames & detections are stored 
 - Query images are also stored inside media/queries/ named using primary key of the query object.
 - Designed to enables rapid sync with S3 or processing via a third party program.

Media directory organization example: 
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
- Pytorch [License](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- Darknet [License](https://github.com/pjreddie/darknet/blob/master/LICENSE)
- AdminLTE2 [License](https://github.com/almasaeed2010/AdminLTE/blob/master/LICENSE)
- FabricJS [License](https://github.com/kangax/fabric.js/blob/master/LICENSE)

# Copyright
Copyright, Akshay Bhat, Cornell University, All rights reserved.