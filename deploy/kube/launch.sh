#!/usr/bin/env bash
set -xe
python create_secrets.py
kubectl create -f secrets.yml
kubectl create -f deployments/gce-pd.yml
kubectl create -f deployments/postgres.yaml
kubectl create -f deployments/rabbitmq.yaml
kubectl create -f deployments/redis.yaml
sleep 60
kubectl create -f deployments/webserver.yaml
sleep 30
kubectl create -f deployments/coco.yaml
kubectl create -f deployments/extractor.yaml
kubectl create -f deployments/face.yaml
kubectl create -f deployments/facenet.yaml
kubectl create -f deployments/facenet_retriever.yaml
kubectl create -f deployments/inception.yaml
kubectl create -f deployments/inception_retriever.yaml
kubectl create -f deployments/global_retriever.yaml
kubectl create -f deployments/textbox.yaml
kubectl create -f deployments/scheduler.yaml
kubectl create -f deployments/crnn.yaml
kubectl create -f deployments/tagger.yaml
