FROM tensorflow/tensorflow:1.2.0-gpu
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>
RUN apt-get update -q && apt-get install -y wget xz-utils
WORKDIR "/bin/"
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
RUN tar xvfJ ffmpeg-release-64bit-static.tar.xz
RUN mv ffmpeg*/* .
WORKDIR "/root/
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ xenial-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN apt-get update && apt-get install -y postgresql-client-9.6 zip
RUN apt-get update -q && apt-get install -y pkg-config python-dev python-opencv unzip libopencv-dev python-pip libav-tools libjpeg-dev  libpng-dev  libtiff-dev  libjasper-dev  python-numpy python-scipy  python-pycurl  python-opencv git nginx supervisor libpq-dev python-cffi build-essential libssl-dev libffi-dev python-dev
RUN dpkg -L python-opencv
RUN pip install --upgrade matplotlib
RUN wget https://yt-dl.org/downloads/latest/youtube-dl -O /bin/youtube-dl
RUN chmod a+rx /bin/youtube-dl
RUN youtube-dl -U