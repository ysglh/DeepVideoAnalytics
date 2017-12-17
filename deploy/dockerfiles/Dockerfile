FROM tensorflow/tensorflow:1.4.0
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN apt-get update -q && apt-get install -y wget xz-utils
WORKDIR "/bin/"
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN tar xvfJ ffmpeg-release-64bit-static.tar.xz
RUN mv ffmpeg*/* .
WORKDIR "/root/"
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ xenial-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN apt-get update && apt-get install -y postgresql-client-9.6 zip
RUN apt-get update -q && apt-get install -y pkg-config python-dev python-opencv unzip libopencv-dev python-pip libav-tools libjpeg-dev  libpng-dev  libtiff-dev  libjasper-dev  python-numpy python-scipy  python-pycurl  python-opencv git nginx supervisor libpq-dev python-cffi build-essential libssl-dev libffi-dev python-dev
RUN dpkg -L python-opencv
RUN pip install --upgrade setuptools
RUN pip install --upgrade matplotlib
RUN wget https://yt-dl.org/downloads/latest/youtube-dl -O /bin/youtube-dl
RUN chmod a+rx /bin/youtube-dl
RUN youtube-dl -U
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
WORKDIR "/root/DVA"
RUN pip install --upgrade cffi
RUN pip install --no-deps keras
RUN pip install -r requirements.txt
RUN pip install pip --upgrade
RUN pip install --no-cache-dir http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
RUN pip install --no-cache-dir mxnet==0.11.0
RUN pip install torchvision
RUN apt-get install -y libhdf5-dev
RUN pip install --no-deps h5py
WORKDIR "/root/DVA"
WORKDIR "/root/DVA/repos/lopq/python"
RUN python setup.py install
WORKDIR "/root/DVA/repos/tf_ctpn_cpu/lib/utils/"
RUN ./make.sh
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80