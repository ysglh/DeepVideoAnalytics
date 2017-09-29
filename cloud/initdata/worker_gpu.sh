#cloud-boothook
#!/bin/sh
set -x
sudo mkdir /efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 {}:/ /efs
sudo chown ubuntu:ubuntu /efs/
cd /efs && mkdir media
service docker restart
docker volume create --opt type=none --opt device=/efs/media --opt o=bind dvadata
cd /home/ubuntu/DeepVideoAnalytics && git pull
sudo pip install --upgrade nvidia-docker-compose awscli
echo 'aws s3 cp s3://aub3config/heroku.env /home/ubuntu/heroku.env && . /home/ubuntu/heroku.env && cd /home/ubuntu/DeepVideoAnalytics/cloud/compose && nvidia-docker-compose -f {} up -d > launch.log 2>error.log &' >> /home/ubuntu/startup.sh
