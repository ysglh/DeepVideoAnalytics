#!/usr/bin/env bash
set -xe
docker build -f Dockerfile.gpu_base -t akshayubhat/dva_gpu_base .
docker push akshayubhat/dva_gpu_base
