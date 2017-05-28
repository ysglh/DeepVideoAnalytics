#!/usr/bin/env bash
set -xe
./rebuild.sh
docker-compose up -d
sleep 600
rm -rf logs
mkdir logs
touch logs/gitkeep
docker cp dva-server:/root/DVA/logs/ logs/
docker-compose down -v