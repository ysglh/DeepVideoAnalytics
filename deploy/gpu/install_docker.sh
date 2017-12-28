#!/usr/bin/env bash
# AMI: Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-37bb714d)
sudo apt-get update
sudo apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get -y install docker-ce=17.09.1~ce-0~ubuntu
sudo usermod -aG docker ubuntu
exit