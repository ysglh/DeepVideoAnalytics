#!/usr/bin/env bash
kubectl create -f gce-pd.yml
kubectl create -f secrets.yml
kubectl create -f postgres.yaml
kubectl create -f rabbitmq.yaml
kubectl create -f webserver.yaml
kubectl create -f extractor.yaml

