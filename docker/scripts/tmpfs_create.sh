#!/usr/bin/env bash
docker volume rm dvadata --force
mkdir /tmp/media
docker volume create --opt type=none --opt device=/tmp/media --opt o=bind dvadata