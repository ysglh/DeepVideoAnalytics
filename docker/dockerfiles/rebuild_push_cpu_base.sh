#!/usr/bin/env bash
set -xe
docker build -f Dockerfile.cpu_base -t akshayubhat/dva_cpu_base .
docker push akshayubhat/dva_cpu_base