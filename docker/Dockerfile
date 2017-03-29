FROM tensorflow/tensorflow:1.0.0

MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN apt-get update && apt-get install -y wget
WORKDIR "/bin/"
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN tar xvfJ ffmpeg-release-64bit-static.tar.xz
RUN mv ffmpeg*/* .
WORKDIR "/root/
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		curl \
		git \
		libffi-dev \
		libssl-dev \
		libtiff5-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-pip \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		libboost-all-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libpq-dev && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*
RUN pip install https://s3.amazonaws.com/pytorch/whl/cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl
RUN pip install scipy
RUN pip install torchvision
RUN pip install --upgrade pip \
                          fabric \
                          django \
                          jinja \
                          jinja2 \
                          djangorestframework \
                          markdown \
                          django-filter \
                          "celery==3.1.23" \
                          "django-celery==3.1.17" \
                          "dj-database-url==0.4.0" \
                          "whitenoise==2.0.6" \
                          raven \
                          psycopg2 \
                          requests \
                          pandas \
                          boto3 \
                          awscli \
                          protobuf \
                          scikit-learn \
                          humanize
RUN apt-get update && apt-get install -y \
    pkg-config \
    python-dev \
    python-opencv \
    libopencv-dev \
    libav-tools  \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    python-numpy \
    python-pycurl \
    python-opencv
RUN wget https://yt-dl.org/downloads/latest/youtube-dl -O /bin/youtube-dl
RUN chmod a+rx /bin/youtube-dl
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
RUN youtube-dl -U
WORKDIR "/root/DVA/darknet"
RUN make
RUN wget https://www.dropbox.com/s/0zopjpswug5rjqy/yolo9000.weights
RUN dpkg -L python-opencv
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN apt-get update && apt-get install -y postgresql-client-9.6 zip
RUN pip install --upgrade matplotlib
WORKDIR "/root/DVA"
RUN git pull
RUN cd dvalib/ssd/checkpoints && unzip ssd_300_vgg.ckpt.zip
RUN apt-get update && apt-get install -y liblapack3 libopenblas-dev
RUN git pull
WORKDIR "/root/DVA/faiss"
RUN make
RUN make py
WORKDIR "/root/DVA"
RUN git pull
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/2unad9skmc7msel/model-20170306-150500.ckpt-250000.data-00000-of-00001 && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/j8ky6bl0jgpygp6/model-20170306-150500.ckpt-250000.index && cd ../../..
RUN cd dvalib/facenet/facenet_model && wget https://www.dropbox.com/s/tnebmpku7xtzwnv/model-20170306-150500.meta && cd ../../..
ENV PYTHONPATH $PYTHONPATH:/root/DVA/faiss
RUN git pull && pip install --upgrade django-crispy-forms lopq
VOLUME ["/root/DVA/dva/media"]
