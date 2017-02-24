## A collection of notes about Deep Video Analytics


#### Creating AWS AMI for Deep Video Analytics

Using [ami-b1e2c4a6](https://blog.empiricalci.com/a-gpu-enabled-ami-for-deep-learning-5aa3d694b630#.m1fse4jvi) as a base ami,
we ran following commands to update cuda (command not shown below), docker engine. Installed docker-compose, nvidia-docker-compose and added ubuntu in group docker.
````bash
sudo apt-get update && sudo apt-get -y install docker-engine python-pip && sudo pip install --upgrade nvidia-docker-compose
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0/nvidia-docker_1.0.0-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
sudo gpasswd -a ${USER} docker && sudo service docker restart
sudo curl -L "https://github.com/docker/compose/releases/download/1.11.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose
git clone https://github.com/AKSHAYUBHAT/deepvideoanalytics && cd deepvideoanalytics/docker_GPU/
````
