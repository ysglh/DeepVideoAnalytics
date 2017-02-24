`````bash
ls
git status
sudo apt-get install --no-install-recommends -y gcc make libc-dev
wget -P /tmp http://us.download.nvidia.com/XFree86/Linux-x86_64/361.42/NVIDIA-Linux-x86_64-361.42.run
sudo apt-get install linux-image-extra-virtual
vim /etc/modprobe.d/blacklist-nouveau.conf
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot
ls
sudo apt-get install linux-source
sudo apt-get install linux-headers-`uname -r`
sudo sh /tmp/NVIDIA-Linux-x86_64-361.42.run --silent
wget -P /tmp http://us.download.nvidia.com/XFree86/Linux-x86_64/361.42/NVIDIA-Linux-x86_64-361.42.run
sudo sh /tmp/NVIDIA-Linux-x86_64-361.42.run --silent
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
echo deb https://apt.dockerproject.org/repo ubuntu-xenial main | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
apt-cache policy docker-engine
docker ps
sudo apt-get install docker-engine
sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
apt-get -f install
sudo apt-get -f install
sudo apt-get install linux-image-extra-$(uname -r) linux-image-extra-virtual
sudo apt-get update
sudo apt-get install docker-engine
sudo apt-get install -f
docker ps
sudo apt-get install docker-engine
sudo service docker start
docker run hello-world
sudo docker run hello-world
sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
nvidia-docker run --rm nvidia/cuda nvidia-smi
sudo nvidia-docker run --rm nvidia/cuda nvidia-smi
ls
ddh -
dh
hd
ls
ls -la /
ls -lah /
dh
ddh
eix
exit
clear
ls
git 
git clone https://github.com/AKSHAYUBHAT/deepvideoanalytics
cd deepvideoanalytics/
pip install --upgrade nvidia-docker-compose
sudo apt-get update && sudo apt install python-pip
sudo apt-get update 
nvidia-smi 
clear
nvidia-docker
clear
lcd
pip
sudo apt install python-pip
tree
top
psaux | grep "root"
ps aux | grep "root"
ps aux | grep "gzip"
sudo apt install python-pip
ls
cd ..
ls
cd .cache/
cd ..
cd .config
ls
ls -a
cat .sudo_as_admin_successful 
cat .wget-hsts 
history
pip install --upgrade nvidia-docker-compose
clear
cd deepvideoanalytics/
ls
clear
cd docker_GPU/
nvidia-docker-compose up
docker ps
docker
docker ps
sudo docker -d
sudo service docker status
clear
docker
docker ps
sudo docker ps
sudo nvidia-docker-compose up
which nvidia-docker-compose 
sudo /home/ubuntu/.local/bin/nvidia-docker-compose up
sudo /home/ubuntu/.local/bin/nvidia-docker-compose 
sudo nvidia-docker
sudo nvidia-docker ls
sudo nvidia-docker ps
sudo pip install nvidia-docker-compose
sudo nvidia-docker-compose up
su
sudo -c
clear
nvidia-docker
docker ps
sudo nvidia-docker-compose up
sudo groupadd docker
sudo gpasswd -a ${USER} docker
sudo service docker restart
docker ps
clear
whoami
clear
exit
docker ps
clear
cd deepvideoanalytics/
ls
cd docker_GPU/
nvidia-docker-compose 
ls
curl -L "https://github.com/docker/compose/releases/download/1.11.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.11.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
clear
sudo chmod +x /usr/local/bin/docker-compose
history
clear
ls
nvidia-docker-compose up
sudo apt-get update
sudo apt-get -y install docker-engine
````