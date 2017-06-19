# Practical Deep OCR for scene text using CTPN + CRNN

This folder cotains two noteboooks which demonstrate use of CTPN (Caffe implementation) [1,2] for
Text box detection and CRNN (PyTorch implmentation) [3,4] for Text character recognition. Most tutorials
online describe traditional OCR techniques using Tessaract. However Tessaract is not useful for scene text recognition, 
i.e. text occuring in natural scenes. Over the last couple of years significant improvements have been made in using 
deep learning for OCR, in this demo we will show how you can use both models to .

To run following two notebooks, clone this repo, start docker container (nvidia-docke/GPU preferred) using following script.
[https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/docker/ctpn/run_ocr_container.sh](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/docker/ctpn/run_ocr_container.sh)

Go to the jupyter notebook url displayed and navigate to notebooks/OCR:

## Text detection

![detection](detection.png "detection")

The textbox detection is implemented using Connectionist Text Proposal Network [1,2].
In this demo images in [images](/notebooks/OCR/images/) folder are processed using CTPN and extracted textboxes are stored in the [boxes](/notebooks/OCR/boxes/)
folder. 

You can find the notebook here
[https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notebooks/OCR/detect_text.ipynb](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notebooks/OCR/detect_text.ipynb)



## Text recognition

![recognition](recognition.png "recognition")

In this notebook the stored boxes are then processed using CRNN [3,4] to extract text. 
Note that you cannot import caffe and pytorch into same notebook/process since it cases library/static linking issues.

You can find the notebook here
[https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notebooks/OCR/recognize_text.ipynb](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/notebooks/OCR/recognize_text.ipynb)


## Integration with Deep Video Analytics

Both CTPN & CRNN have been integrated into [Deep Video Analytics](https://www.deepvideoanalytics.com) and now its possible to run OCR directly on videos/images
without having to write any code. CTPN and CRNN run as tasks on a celery queue named "qocr". Workers consuming qocr need to run on
dva_ctpn container image. This [OCR docker-compose](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics/blob/master/docker/custom_compose/docker-compose-gpu-ocr.yml) file describes the setup. The extracted bounding boxes and text are 
represented as Regions. Further extracted text can be conveniently queried using Postgres full-text search through the User Interface.



### References:

1. [https://github.com/tianzhi0549/CTPN](https://github.com/tianzhi0549/CTPN)
2. [https://github.com/qingswu/CTPN (CUDA 8.0 compatible, used here)](https://github.com/qingswu/CTPN)
3. [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
4. [https://github.com/bgshih/crnn (Original implementation)](https://github.com/bgshih/crnn)
