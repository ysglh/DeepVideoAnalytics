FROM akshayubhat/dva_cpu_base
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
RUN youtube-dl -U
WORKDIR "/root/DVA/darknet"
RUN pip install -r requirements.txt
RUN make
RUN wget https://www.dropbox.com/s/0zopjpswug5rjqy/yolo9000.weights
WORKDIR "/root/DVA"
RUN cd dvalib/ssd/checkpoints && unzip ssd_300_vgg.ckpt.zip
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/2unad9skmc7msel/model-20170306-150500.ckpt-250000.data-00000-of-00001 && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/j8ky6bl0jgpygp6/model-20170306-150500.ckpt-250000.index && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/tnebmpku7xtzwnv/model-20170306-150500.meta && cd ../../..
RUN cd dvalib/data/2016_08 && wget https://www.dropbox.com/s/umo0xtljm9an90x/model.ckpt && wget https://www.dropbox.com/s/f93axdtlb3ltj40/model.ckpt.meta && cd ../../..
WORKDIR "/root/DVA/repos/lopq/python"
RUN python setup.py install
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80