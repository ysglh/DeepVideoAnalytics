# Schema, Architecture & Backups 
**(Currently writing)**

### App server/workers, db, queue server and volumes 
![Architecture](architecture.png "architecture")




### Execution of celery tasks and data 
![Flowchart](flowchart.png "flowchart")


## Deploying on multiple machines
As an example consider the case where you have
- Webserver on Heroku
- Running a sepearte RabbitMQ & Postgres is easy since docker container uses environment variables to connect.

### Sharing media folder

Two solutions:

#### 1. Using NFS or AWS EFS (slow without prefetching)

#### 2. Using AWS S3 and prefetching images

### Re-routing requests for /media/

Ideally django filter etc. when creating those urls within templates.

Temporary solution simply redirect to S3 bucket.

#Database and Folder Schema


## Media folder schema

 - One directory per video or dataset (a set of images)
 - Extracted frames and detections are stored in regions/ & frames/ under the video directory
 - Indexes (numpy arrays) and list of corresponding frames & detections are stored 
 - Query images are also stored inside media/queries/ named using primary key of the query object.
 - Designed to enables rapid sync with S3 or processing via a third party program.

###Media directory organization example: 
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

## Database schema

###Tables
1. video
2. frame
3. detection
4. query
5. framelable
6. queryresult
7. TEvents
