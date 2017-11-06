#!/usr/bin/env bash
source ~/aws.env
docker-compose -f docker-compose-linode.yml down -v
set -xe
docker-compose -f docker-compose-linode.yml up -d
sleep 100
docker exec -u="root" -it dva-server bash -c "fab superu"
