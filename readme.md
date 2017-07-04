# Deep Video Analytics  •  [![Build Status](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics.svg?branch=master)](https://travis-ci.org/AKSHAYUBHAT/DeepVideoAnalytics)

![UI Screenshot](notebooks/images/emma.png "Emma Watson, from poster of her latest subject appropriate movie The Circle")
![Banner](notebooks/images/banner_small.png "banner")


**Don't be worried by complexity of this banner, with latest version of docker installed correctly, you can run Deep Video Analytics in minutes locally (even without a GPU) using a single command.** 

#### Author: [Akshay Bhat, Cornell University.](http://www.akshaybhat.com)

#### Deep Video Analytics is a platform for indexing and extracting information from videos and images. For installation instructions & demo go to [https://www.deepvideoanalytics.com](https://www.deepvideoanalytics.com)

#### Libraries used/modified in code and their licenses

| Library  | Link to the license | 
| -------- | ------------------- |
| YAD2K  |  [MIT License](https://github.com/allanzelener/YAD2K/blob/master/LICENSE)  |
| AdminLTE2  |  [MIT License](https://github.com/almasaeed2010/AdminLTE/blob/master/LICENSE) |
| FabricJS |  [MIT License](https://github.com/kangax/fabric.js/blob/master/LICENSE)  |
| Facenet   |  [MIT License](https://github.com/davidsandberg/facenet)  |
| JSFeat   |  [MIT License](https://inspirit.github.io/jsfeat/)  |
| MTCNN   |  [MIT License](https://github.com/kpzhang93/MTCNN_face_detection_alignment)  |
| CRNN.pytorch  |  [MIT License](https://github.com/meijieru/crnn.pytorch/blob/master/LICENSE.md)  |
| Original CRNN code by Baoguang Shi  |  [MIT License](https://github.com/bgshih/crnn) |
| Object Detector App using TF Object detection API |  [MIT License](https://github.com/datitran/Object-Detector-App) | 
| Plotly.js |  [MIT License](https://github.com/plotly/plotly.js/blob/master/LICENSE) | 
| Segment annotator  |   [BSD 3-clause](https://github.com/kyamagu/js-segment-annotator/blob/master/LICENSE) |
| TF Object detection API  | [Apache 2.0](https://github.com/tensorflow/models/tree/master/object_detection) |
| LOPQ   |  [Apache 2.0](https://github.com/yahoo/lopq/blob/master/LICENSE)  | 
| Open Images Pre-trained network |  [Apache 2.0](https://github.com/openimages/dataset/blob/master/LICENSE) |

#### Additionally following libraries & frameworks are installed when building/running the container

* FFmpeg (not linked, called via a Subprocess)
* Tensorflow 
* OpenCV
* Numpy
* Pytorch
* Docker
* Nvidia-docker
* Docker-compose
* All packages in [requirements.txt](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/requirements.txt) & used in Dockerfiles.

### Data & processing model
![Data model](notebooks/images/infographic.png "graphic")


# License & Copyright

**Copyright 2016-2017, Akshay Bhat, Cornell University, All rights reserved.**

# Contact

Deep Video Analytics is currently in active development.
The license will be relaxed once the a stable release version is reached.
Please contact me for more information. For more information see [answer on this issue](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/issues/29)
 
