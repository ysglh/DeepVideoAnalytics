#!/usr/bin/env bash
kubectl delete -f secrets.yml
kubectl delete -f deployments/coco.yaml
kubectl delete -f deployments/extractor.yaml
kubectl delete -f deployments/face.yaml
kubectl delete -f deployments/facenet.yaml
kubectl delete -f deployments/facenet_retriever.yaml
kubectl delete -f deployments/inception.yaml
kubectl delete -f deployments/inception_retriever.yaml
kubectl delete -f deployments/postgres.yaml
kubectl delete -f deployments/rabbitmq.yaml
kubectl delete -f deployments/textbox.yaml
kubectl delete -f deployments/webserver.yaml