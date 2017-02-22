#!/usr/bin/env bash
set -xe
./rebuild.sh
nvidia-docker-compose up -d
sleep 600
rm -rf logs
mkdir logs
touch logs/gitkeep
nvidia-docker cp dva-server:/root/DVA/logs/ logs/
nvidia-docker-compose down -v