# Towards large scale intelligent systems that learn continuously and interactively in a reproducible manner


### State of AI/ML libraries & framework
Over the last five years with emergence of deep learning, several libraries/frameworks such as Caffe, Torch, 
PyTorch, Tensor Flow have become popular. With each new library solving problems such as model portability,
distributed training, autograd, etc. For more information about evolution of Deep Learning Libraries listen 
to this podcast by Soumith (one of the creator or PyTorch).

![modelcentric](figures/modelcentric.png "model centric")

Libraries such as PyTorch, Caffe and TensorFlow, only specify model and training/inference. A significant part
Computer Vision research and applications involves data collection, annotation, organization. Currently there 
are no frameworks that support these tasks. The closest equivalent would be something like Robot Operating System (ROS)
in robotics.  

### Towards a data-centric approach
Currently most of the published research in Computer Vision uses a model-centric pattern illustrated above. 
Typically researchers download existing dataset or collect & annotate new data. Either a baseline model or
 a new model is proposed for solving a particular problem (detection, segmentation, counting etc.) 
 While the pattern described above works well for specific tasks, it slows the research in Computer Vision, 
 especially in topics such as interactive learning or learning by continuously ingesting data.

### System architecture for distributed intelligence
![system](figures/system.png "Ideal system")

  
### Practical scaling with containers, spot/premptible and lambda
 ![cloud](figures/cloud.png "distributed architecture")

### A vision for future: From yearly competitions to near-realtime collaboration 

Today yearly competitions such as COCO, ILSVRC are the main form of establishing state of the art. 
 