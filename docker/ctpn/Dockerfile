FROM nvidia/cuda:7.0-cudnn3-devel
MAINTAINER Akshay Bhat <akshayubhat@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        zip \
        unzip \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CTPN_ROOT=/opt/ctpn
WORKDIR $CTPN_ROOT

RUN git clone --depth 1 https://github.com/tianzhi0549/CTPN.git
WORKDIR $CTPN_ROOT/CTPN/caffe

# Missing "packaging" package
RUN pip install --upgrade pip
RUN pip install packaging

RUN cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd ..
WORKDIR /

WORKDIR $CTPN_ROOT/CTPN/caffe
RUN cp Makefile.config.example Makefile.config
RUN mkdir build && cd build && cmake -DUSE_CUDNN=1 .. && WITH_PYTHON_LAYER=1 make -j"$(nproc)" && make pycaffe
ENV PYCAFFE_ROOT $CTPN_ROOT/CTPN/caffe/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CTPN_ROOT/CTPN/caffe/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CTPN_ROOT/CTPN/caffe/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
RUN cp $CTPN_ROOT/CTPN/src/layers/* $CTPN_ROOT/CTPN/caffe/src/caffe/layers/
RUN cp $CTPN_ROOT/CTPN/src/*.py $CTPN_ROOT/CTPN/caffe/src/caffe/
RUN cp -r $CTPN_ROOT/CTPN/src/utils $CTPN_ROOT/CTPN/caffe/src/caffe/
RUN cd ~ && mkdir -p ocv-tmp && cd ocv-tmp && wget https://github.com/Itseez/opencv/archive/2.4.12.zip
RUN cd ~/ocv-tmp && unzip 2.4.12.zip && cd opencv-2.4.12 && mkdir release
RUN cd ~/ocv-tmp/opencv-2.4.12/release && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_PYTHON_SUPPORT=ON .. && make -j8 && make install && rm -rf ~/ocv-tmp
WORKDIR $CTPN_ROOT/CTPN
RUN make
RUN pip install --upgrade numpy
RUN wget "https://www.dropbox.com/s/yyj53aet2emhvs7/ctpn_trained_model.caffemodel?dl=1" -O $CTPN_ROOT/CTPN/models/ctpn_trained_model.caffemodel
RUN mkdir /opt/ctpn/CTPN/output
VOLUME ['/opt/ctpn/CTPN/output/']
RUN pip install --upgrade jupyter
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
WORKDIR /opt/ctpn/CTPN/
EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--allow-root"]
RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ trusty-pgdg main" > /etc/apt/sources.list.d/pgdg.list
RUN apt-get update && apt-get install -y postgresql-client-9.6 zip libpq-dev libssl-dev
RUN git clone https://github.com/akshayubhat/DeepVideoAnalytics /root/DVA
WORKDIR "/root/DVA"
RUN pip install --upgrade cffi
RUN pip install -r requirements.txt
VOLUME ["/root/DVA/dva/media"]