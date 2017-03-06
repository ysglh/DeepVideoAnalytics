#Deep Video Analytics  •  [![Build Status](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics.svg?branch=master)](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics)
![Banner](notes/banner_small.png "banner")

#### Author: [Akshay Bhat, Cornell University.](http://www.akshaybhat.com)       


Deep Video Analytics provides a platform for indexing and extracting information from videos and images.
Deep learning detection and recognition algorithms are used for indexing individual frames / images along with 
detected objects. The goal of Deep Video analytics is to become a quickly customizable platform for developing 
visual & video analytics applications, while benefiting from seamless integration with state or the art models released
by the vision research community.

##### self-promotion: If you are interested in Healthcare & Machine Learning please take a look at my another Open Source project [Computational Healthcare](http://www.computationalhealthcare.com) 

## Features
- Visual Search using Nearest Neighbors algorithm as a primary interface
- Upload videos, multiple images (zip file with folder names as labels)
- Provide Youtube url to be automatically processed/downloaded by youtube-dl
- Metadata stored in Postgres
- Operations (Querying, Frame extraction & Indexing) performed using celery tasks and RabbitMQ
- Separate queues and workers for selection of machines with different specifications (GPU vs RAM) 
- Videos, frames, indexes, numpy vectors stored in media directory, served through nginx
- Explore data, manually run code & tasks without UI via a jupyter notebook [explore.ipynb](experiments/Notebooks/explore.ipynb)
- [Some documentation on design decision, architecture and deployment](/notes/readme.md).

## Several models included out of the box
**We take significant efforts to ensure that following models (code+weights included) work without having to write any code.**

- [x] Indexing using Google inception V3 trained on Imagenet
- [x] [Single Shot Detector (SSD) Multibox 300 training using VOC](https://github.com/balancap/SSD-Tensorflow)  
- [x] Alexnet using Pytorch  (disabled by default; set ALEX_ENABLE=1 in environment variable to use)
- [x] [YOLO 9000](http://pjreddie.com/darknet/yolo/) (disabled by default; set YOLO_ENABLE=1 in environment variable to use)
- [X] [Face detection/alignment/recognition using MTCNN and Facenet](https://github.com/davidsandberg/facenet) 
- [ ] [Facebook FAISS for fast approximate similarity search (Coming very soon!)]()

## Potential algorithms/models 
1. [ ] [Text detection models](http://www.robots.ox.ac.uk/~vgg/research/text/)
2. [ ] [Soundnet (requires extracting mp3 audio)](http://projects.csail.mit.edu/soundnet/)
3. [ ] [Pytorch Squeezenet](http://pytorch.org/docs/torchvision/models.html)
4. [ ] [Mapnet (requires converting models from Marvin)](http://www.cs.princeton.edu/~aseff/mapnet/)   
5. [ ] [Keras-js](https://github.com/transcranial/keras-js) which uses Keras inception for client side indexing   

### Open Issues & To Do

- [ ] [OpenCV fails with double free error at cap.release() on some ubuntu machines in spite of running in a docker container](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/issues/4)
- [ ] [When running SSD-TensorFlow in a celery task, the code aborts during second task, as a result its reloaded every time via a subprocess](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/issues/3)
- [Please take a look at this board for planned future tasks](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/projects/1)


## Installation
**Pre-built docker images (corrosponding to alpha version) for both CPU and GPU version are now [available on Docker Hub](https://hub.docker.com/r/akshayubhat/dva/tags/).** 

### On Mac, Windows and Linux machines without NVidia GPUs
You need to have latest version of Docker installed.
````bash
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics && cd DeepVideoAnalytics/docker && docker-compose up 
````

### Your machine NVidia GPU with Docker and nvidia-docker installed
Replace docker-compose by nvidia-docker-compose, the Dockerfile uses tensorflow gpu base image and appropriate version of pytorch. The Makefile for Darknet is also modified accordingly. This code was tested using an older NVidia Titan GPU and nvidia-docker.

```bash
pip install --upgrade nvidia-docker-compose
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics && cd DeepVideoAnalytics/docker_GPU && ./rebuild.sh && nvidia-docker-compose up 
```
### On AWS EC2 with a GPU enabled P2 instance
We provide an AMI will all dependancies such as docker & drivers. Start a P2.xlarge instance with **ami-b3cc1fa5** (N. Virginia), ports 8000, 6006, 8888 open (preferably to only your IP) and run following command after ssh'ing into the machine. 
```bash
cd deepvideoanalytics && git pull && cd docker_GPU && ./rebuild.sh && nvidia-docker-compose up 
```
you can optionally specify "-d" at the end to detach it, but for the very first time its useful to read how each container is started. After approximately 3 ~ 5 minutes the user interface will appear on port 8000 of the instance ip.
The Process used for [AMI creation is here](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notes/ami.md) **Security warning!** The current GPU container uses nginx <-> uwsgi <-> django setup to ensure smooth playback of videos. 
However it runs nginix as root (though within the container). Considering that you can now modify AWS Security rules on-the-fly, I highly recommend allowing inbound traffic only from your own IP address.

### On multiple machines with/without GPUs
Other than the shared media folder (ideally a mounted EFS or NFS), configuring Postgres and RabbitMQ is straightforward.
Please [read this regarding trade offs](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notes/architecture.md).

#### Options specified via environment variable
Following options can be specified in docker-compose.yml, or your envrionment.

- ALEX_ENABLE=1 (to use Alexnet with PyTorch. Otherwise disabled by default)
- YOLO_ENABLE=1 (to use YOLO 9000. Otherwise disabled by default)
- SCENEDETECT_DISABLE=1 (to disable scene detection, Otherwise enabled by default)

 
## Architecture
![Architecture](notes/architecture.png "System architecture")

## User Interface 
### Search
![UI Screenshot](notes/search.png "search")
### Past queries
![UI Screenshot](notes/past_query.png "past queries")
### Video list / detail
![UI Screenshot](notes/video_list.png "Video list")
![UI Screenshot](notes/video_detail.png "detail")
### Frame detail
![UI Screenshot](notes/frame_detail.png "Frame detail")

## Libraries & Code used

- Pytorch [License](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- Darknet [License](https://github.com/pjreddie/darknet/blob/master/LICENSE)
- AdminLTE2 [License](https://github.com/almasaeed2010/AdminLTE/blob/master/LICENSE)
- FabricJS [License](https://github.com/kangax/fabric.js/blob/master/LICENSE)
- Modified PySceneDetect [License](https://github.com/Breakthrough/PySceneDetect)
- Modified SSD-Tensorflow [Individual files are marked as Apache](https://github.com/balancap/SSD-Tensorflow)
- FAISS [License (Non Commercial)](https://github.com/facebookresearch/faiss)
- Facenet [License](https://github.com/davidsandberg/facenet)
- MTCNN [TensorFlow port of MTCNN for face detection/alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- Docker 
- Nvidia-docker
- OpenCV
- Numpy
- FFMPEG
- Tensorflow

## References


## Citation for Deep Video Analytics

**Coming soon!**

# Copyright
**Copyright 2016-2017, Akshay Bhat, Cornell University, All rights reserved.**
