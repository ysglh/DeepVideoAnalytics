#!/usr/bin/env bash
set -xe
./rebuild.sh
docker-compose up -d
sleep 600
rm -rf test_logs
mkdir test_logs
touch test_logs/gitkeep
docker cp dva-server:/root/DVA/logs/ test_logs/
docker-compose down -v