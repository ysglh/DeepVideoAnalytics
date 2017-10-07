# Towards large scale intelligent systems that learn continuously and interactively

| **"we shape our buildings thereafter they shape us"** -- Winston Churchil

Over the last five years with emergence of Deep Learning, several libraries and frameworks such as Caffe, Torch, 
PyTorch, MXNet & TensorFlow have been developed. Each new library has solved new problems such as model portability,
distributed training, autograd, etc. However they are developed primarily with goal of training models individual tasks. 
We believe this model-centric approach is not only slowing down research but also limiting possibility of novel intelligent 
systems. As a result we propose a data-centric approach which standardizes data and processing in addition to models. We present 
Deep Video Analytics a distributed visual data processing system which adopts a data-centric approach. 

![modelcentric](figures/modelcentric.png "model centric")
 
 ### Deep Video Analytics: A distributed visual data processing system 
 Today most the published research in Computer Vision uses a model-centric pattern illustrated above. 
 Typically researchers download existing dataset or collect & annotate new data. Either a baseline model or
 a new model is proposed and evaluated for solving a particular problem (detection, segmentation, counting etc.)
 A significant part Computer Vision research and applications involves data collection, annotation and organization. Currently there 
 are no frameworks that support these tasks. The closest would be Robot Operating System (ROS) used in robotics.  
 While the pattern described above works well for evaluation of specific tasks, it slows down research in Computer Vision, 
 especially in topics such as interactive learning or learning by continuously ingesting data. We believe with Deep Video 
 Analytics we can enable novel research as well as applications that are even difficult to imagine in the current
 model-centric paradigm. 
 
![datacentric](figures/datacentric.png "data centric")

### Architecture
![system](figures/system.png "Ideal system")

 
#### Three essential components: Language, Data model & Implementation
To build a distributed visual data processing system we provide three components which we believe are essential to build the system
in an extensible manner.

- **Data & Processing model**: We provide schema for representing videos, frames, regions, tubes, models and events.
- **Language**: Similar to SQL we introduce DVAPQL an event based language for ingesting, processing and 
  querying (via visual search) visual data.
- **Implementation**: We provide a function implementation of the above two that scales to multiple machines and includes a
  user interface & APIs.

We hope that by working with vision research community we can iterate on data model and language, while improving the implementation.

 
#### Practical scaling with containers, spot/preemptible and lambda
![cloud](figures/cloud.png "Distributed architecture")
 
With advent of cloud computing, its now possible to launch thousands of short lived tasks to
10~100s of instances with GPUs. Building systems capable of leveraging this type of computing power
 is not trivial, further until now such systems required significant amount of DevOPs / Systems Administration.
 However with advent of containers which abstract away all OS level components such as libraries, drivers etc. and
 container orchestration systems such as Kubernetes (supported by Google Container Engines) and AWS Elastic Container 
 Service. It is now possible to deterministically specify all components. 
 
### A Brave New World of possibilities 

#### Reproducible research in interactive & continuous learning
 
Several researcher have proposed systems that learn continuously by surfing web or through user feedback.
However these studies typically do not share code used for building the system, even if the code is shared 
its not easily extensible.  

#### From annual competitions to near-real-time collaboration

Today yearly competitions such as COCO, ILSVRC are the gold standard for establishing state of the art.