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
RUN pip install pip --upgrade
RUN pip install --no-cache-dir http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
RUN pip install --no-cache-dir mxnet==0.11.0
RUN pip install torchvision
WORKDIR "/root/DVA"
RUN cd dvalib/crfasrnn/cpp && ./compile.sh && cd ../../..
WORKDIR "/root/DVA/repos/lopq/python"
RUN python setup.py install
WORKDIR "/root/DVA"
VOLUME ["/root/DVA/dva/media"]
EXPOSE 80