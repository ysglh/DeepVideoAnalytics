# Deep Video Analytics  •  [![Build Status](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics.svg?branch=master)](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics)

![Banner](notes/banner_small.png "banner")
![UI Screenshot](notes/face_recognition.png "face recognition")
#### Author [Akshay Bhat, Cornell University.](http://www.akshaybhat.com)       

### [Visit the website to watch demo video of the user interface and for installation instructions](https://deepvideoanalytics.com)

Deep Video Analytics provides a platform for indexing and extracting information from videos and images.
Deep learning detection and recognition algorithms are used for indexing individual frames/images along with 
detected objects. The goal of Deep Video analytics is to become a quickly customizable platform for developing 
visual & video analytics applications, while benefiting from seamless integration with state or the art models & datasets
released by the vision research community. 

## Features
- Visual Search using Nearest Neighbors algorithm as a primary interface
- Upload videos, multiple images (zip file with folder names as labels)
- Provide Youtube url to be automatically processed/downloaded by youtube-dl
- Leverage pre-trained object recognition/detection, face recognition models for analysis and visual search.
- Query against pre-indexed external datasets containing millions of images.
- Metadata stored in Postgres, Operations performed asynchronously using celery tasks. 
- Separate queues and workers for selection of machines with different specifications (GPU vs RAM).
- Videos, frames, indexes, numpy vectors stored in media directory, served through nginx
- Explore data, manually run code & tasks without UI via a jupyter notebook [explore.ipynb](experiments/Notebooks/explore.ipynb)

##### self-promotion: If you are interested in Healthcare & Machine Learning please take a look at my another Open Source project [Computational Healthcare](http://www.computationalhealthcare.com)

## Models included out of the box
**We take significant efforts to ensure that following models (code+weights included) work without having to write any code.**

- [x] Indexing using Google inception V3 trained on Imagenet
- [x] [Single Shot Detector (SSD) Multibox 300 training using VOC](https://github.com/balancap/SSD-Tensorflow)  
- [x] Alexnet using Pytorch  (disabled by default; set ALEX_ENABLE=1 in environment variable to use)
- [x] [YOLO 9000](http://pjreddie.com/darknet/yolo/) (disabled by default; set YOLO_ENABLE=1 in environment variable to use)
- [x] [Face detection/alignment/recognition using MTCNN and Facenet](https://github.com/davidsandberg/facenet) 

## External datasets indexed for use

- [ ] [Product images data (coming soon!)](http://www.product-open-data.com/download/)
- [ ] [Visual Genome (coming soon!)](http://visualgenome.org/)

## Planned models and datasets

- [ ] [MultiNet or KittiBox](https://github.com/MarvinTeichmann/MultiNet)
- [ ] [Text detection models](http://www.robots.ox.ac.uk/~vgg/research/text/)
- [ ] [Soundnet (requires extracting mp3 audio)](http://projects.csail.mit.edu/soundnet/)
- [ ] [Open Images dataset pretrained inception v3](https://github.com/openimages/dataset)   
- [ ] [Keras-js](https://github.com/transcranial/keras-js) which uses Keras inception for client side indexing   

## Approximate Nearest Neighbors indexing algorithms

- [ ] [Yahoo/Flickr Locally Optimized Product Quantization (currently being integrated for querying external data)](https://github.com/yahoo/lopq)
- [ ] [Facebook FAISS for fast approximate similarity search (lower priority)](https://github.com/facebookresearch/faiss)


## To Do
[Please take a look at this board for planned future tasks](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/projects/1)


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

# References

1. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
2. Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
3. Zhang, Kaipeng, et al. "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." IEEE Signal Processing Letters 23.10 (2016): 1499-1503.
4. Liu, Wei, et al. "SSD: Single shot multibox detector." European Conference on Computer Vision. Springer International Publishing, 2016.
5. Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
6. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.	
7. Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity search with GPUs." arXiv preprint arXiv:1702.08734 (2017).

# Citation 

**Citation for Deep Video Analytics coming soon.**

# Copyright

**Copyright 2016-2017, Akshay Bhat, Cornell University, All rights reserved.**


Please contact me for more information, I plan on relaxing the license soon, once a beta version is reached 
(To the extent allowed by the code/models included.e.g. FAISS disallows commercial use.). 
 
