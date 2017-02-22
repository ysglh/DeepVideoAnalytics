#!/usr/bin/env bash
set -xe
./rebuild.sh
nvidia-docker-compose up -d
sleep 360
rm test_logs/*.log
#docker exec -u="root" -it dva-server bash
nvidia-docker cp dva-server:/root/DVA/logs/ test_logs/
#nvidia-docker-compose down -v