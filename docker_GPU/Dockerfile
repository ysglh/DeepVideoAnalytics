FROM tensorflow/tensorflow:0.11.0-gpu

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
RUN wget https://s3.amazonaws.com/pytorch/whl/cu80/torch-0.1.7.post2-cp27-none-linux_x86_64.whl 
RUN pip install torch-0.1.7.post2-cp27-none-linux_x86_64.whl
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
                          protobuf \
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
RUN youtube-dl -U
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
WORKDIR "/root/DVA/darknet"
RUN mv Makefile.gpu Makefile
RUN make
RUN wget https://www.dropbox.com/s/0zopjpswug5rjqy/yolo9000.weights
VOLUME ["/root/DVA/dva/media"]
WORKDIR "/root/DVA"
