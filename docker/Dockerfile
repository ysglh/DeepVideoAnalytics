FROM ubuntu:14.04
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN apt-get update && apt-get install -y wget xz-utils
WORKDIR "/bin/"
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN tar xvfJ ffmpeg-release-64bit-static.tar.xz
RUN mv ffmpeg*/* .
WORKDIR "/root/
RUN apt-get update -q && apt-get install -y pkg-config python-dev python-opencv unzip libopencv-dev python-pip libav-tools libjpeg-dev  libpng-dev  libtiff-dev  libjasper-dev  python-numpy python-scipy  python-pycurl  python-opencv
RUN dpkg -L python-opencv
RUN wget https://yt-dl.org/downloads/latest/youtube-dl -O /bin/youtube-dl
RUN chmod a+rx /bin/youtube-dl
RUN youtube-dl -U
RUN apt-get update -q && apt-get install -y git
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
RUN youtube-dl -U
WORKDIR "/root/DVA/darknet"
RUN make
RUN wget https://www.dropbox.com/s/0zopjpswug5rjqy/yolo9000.weights
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN apt-get update && apt-get install -y postgresql-client-9.6 zip
RUN pip install --upgrade matplotlib
WORKDIR "/root/DVA"
RUN git pull
RUN cd dvalib/ssd/checkpoints && unzip ssd_300_vgg.ckpt.zip
RUN apt-get update && apt-get install -y libpq-dev python-cffi build-essential libssl-dev libffi-dev python-dev
RUN pip install --upgrade cffi "celery==3.1.23"
RUN pip install -r requirements.txt
#RUN apt-get update && apt-get install -y liblapack3 libopenblas-dev
#RUN git pull
#WORKDIR "/root/DVA/faiss"
#RUN make
#RUN make py
# pip install --upgrade lopq
#ENV PYTHONPATH $PYTHONPATH:/root/DVA/faiss
#WORKDIR "/root/DVA"
RUN git pull
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/2unad9skmc7msel/model-20170306-150500.ckpt-250000.data-00000-of-00001 && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/j8ky6bl0jgpygp6/model-20170306-150500.ckpt-250000.index && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/tnebmpku7xtzwnv/model-20170306-150500.meta && cd ../../..
RUN git pull && pip install --upgrade django-crispy-forms
RUN apt-get update && apt-get install -y nginx supervisor
RUN pip install --upgrade uwsgi
WORKDIR "/root/DVA/repos/lopq/python"
RUN pip install --upgrade lmdb
RUN python setup.py install
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80