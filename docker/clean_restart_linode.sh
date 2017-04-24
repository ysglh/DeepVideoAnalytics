#!/usr/bin/env bash
docker-compose -f docker-compose-linode.yml down -v
docker-compose -f docker-compose-linode.yml up -d
sleep 120
docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli"
docker cp ~/.aws dva-server:/root/.aws