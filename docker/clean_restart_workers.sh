#!/usr/bin/env bash
source heroku.env
docker-compose -f custom_compose/docker-compose-worker.yml down -v
docker-compose -f custom_compose/docker-compose-worker.yml up -d
sleep 120
docker exec -u="root" -it dva-server bash -c "fab superu"
docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli"
docker cp ~/.aws dva-server:/root/.aws