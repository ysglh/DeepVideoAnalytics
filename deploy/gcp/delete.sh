#!/usr/bin/env bash
kubectl delete -f gce-pd.yml
kubectl delete -f secrets.yml
kubectl delete -f postgres.yaml
kubectl delete -f rabbitmq.yaml
kubectl delete -f webserver.yaml

