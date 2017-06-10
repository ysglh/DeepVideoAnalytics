#!/usr/bin/env bash
docker volume create --opt type=none --opt device=$1 --opt o=bind dvadata
nvidia-docker-compose -f custom_compose/docker-compose-gpu-dev.yml up -d