#!/usr/bin/env bash
source ~/aws.env && docker-compose -f docker-compose-linode-non-nfs.yml down -v
source ~/aws.env && docker-compose -f docker-compose-linode-non-nfs.yml up -d
sleep 100
docker exec -u="root" -it webserver bash -c "fab superu"
