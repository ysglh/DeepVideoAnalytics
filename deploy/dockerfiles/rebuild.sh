#!/usr/bin/env bash
set -xe
docker build -t akshayubhat/dva-auto:latest .
docker build -t akshayubhat/dva-auto:gpu -f Dockerfile.gpu .