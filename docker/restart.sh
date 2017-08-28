#!/usr/bin/env bash
docker-compose -f custom/docker-compose-linode.yml down
docker-compose -f custom/docker-compose-linode.yml up -d
sleep 120
docker exec -u="root" -it dva-server bash -c "fab superu"
docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli"
docker cp ~/.aws dva-server:/root/.aws
