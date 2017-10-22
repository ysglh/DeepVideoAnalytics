#!/usr/bin/env bash
docker-compose -f docker-compose-linode.yml down -v
set -xe
docker-compose -f docker-compose-linode.yml up -d
sleep 20
docker cp ~/.aws dva-server:/root/.aws
docker cp ~/.aws dva-caffe:/root/.aws
sleep 100
docker exec -u="root" -it dva-server bash -c "fab superu"
docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli"
