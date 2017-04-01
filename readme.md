# Deep Video Analytics  •  [![Build Status](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics.svg?branch=master)](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics)

![Banner](notes/banner_small.png "banner")
![UI Screenshot](notes/face_recognition.png "face recognition")
#### Author [Akshay Bhat, Cornell University.](http://www.akshaybhat.com)       

### [Visit the website for demo video of the UI and installation instructions](https://deepvideoanalytics.com)

Deep Video Analytics provides a platform for indexing and extracting information from videos and images.
Deep learning detection and recognition algorithms are used for indexing individual frames/images along with 
detected objects. The goal of Deep Video analytics is to become a quickly customizable platform for developing 
visual & video analytics applications, while benefiting from seamless integration with state or the art models & datasets
released by the vision research community. 

**We take significant efforts to ensure that following models (code+weights included) work without having to write any code.**

- [x] Indexing using Google inception V3 trained on Imagenet
- [x] [Single Shot Detector (SSD) Multibox 300 training using VOC](https://github.com/balancap/SSD-Tensorflow)  
- [x] Alexnet using Pytorch  (disabled by default; set ALEX_ENABLE=1 in environment variable to use)
- [x] [YOLO 9000](http://pjreddie.com/darknet/yolo/) (disabled by default; set YOLO_ENABLE=1 in environment variable to use)
- [x] [Face detection/alignment/recognition using MTCNN and Facenet](https://github.com/davidsandberg/facenet) 

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
- Locally Optimized Product Quantization [License](https://github.com/yahoo/lopq/blob/master/LICENSE)
- Docker 
- Nvidia-docker
- OpenCV
- Numpy
- FFMPEG
- Tensorflow

# License & Copyright

**Copyright 2016-2017, Akshay Bhat, Cornell University, All rights reserved.**


Please contact me for more information, I plan on relaxing the license soon, once a beta version is reached 
(To the extent allowed by the code/models included.e.g. FAISS disallows commercial use.). 
 
