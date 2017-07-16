FROM akshayubhat/dva_cpu_base
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN youtube-dl -U
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
WORKDIR "/root/DVA"
RUN pip install --upgrade cffi
RUN pip install --no-deps keras
RUN pip install -r requirements.txt
RUN apt-get install -y libhdf5-dev
RUN pip install --no-deps h5py
WORKDIR "/root/DVA"
RUN cd dvalib/data && wget https://www.dropbox.com/s/jytpgw8et09ede9/facenet.pb && cd ../..
RUN cd dvalib/data && wget https://www.dropbox.com/s/jytpgw8et09ede9/facenet.pb && cd ../..
RUN cd dvalib/data/2016_08 && wget https://www.dropbox.com/s/umo0xtljm9an90x/model.ckpt && wget https://www.dropbox.com/s/f93axdtlb3ltj40/model.ckpt.meta && cd ../../..
RUN cd dvalib/yolo/model_data && wget https://www.dropbox.com/s/zbff2rkoejx5k5r/yolo.h5
WORKDIR "/root/DVA/repos/lopq/python"
RUN python setup.py install
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80