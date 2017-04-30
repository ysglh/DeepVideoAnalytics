#!/usr/bin/env bash
set -xe
cd dva_cpu_base && docker build -t akshayubhat/dva_cpu_base . && cd ..
docker login
docker push akshayubhat/dva_cpu_base
docker build -t akshayubhat/dva:cpu .