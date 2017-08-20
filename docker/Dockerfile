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
RUN cd dvalib/crfasrnn/cpp && ./compile.sh && cd ../../..
WORKDIR "/root/DVA/repos/lopq/python"
RUN python setup.py install
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80