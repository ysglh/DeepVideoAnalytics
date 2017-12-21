import os
import config

command = 'gcloud beta container --project "{project_name}" clusters create "{cluster_name}" ' \
          '--zone "{zone}" ' \
          '--username="admin" --cluster-version "1.7.8-gke.0" --machine-type "{machine_type}" --image-type "COS" ' \
          '--disk-size "100" --scopes "https://www.googleapis.com/auth/compute",' \
          '"https://www.googleapis.com/auth/devstorage.read_write",' \
          '"https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring",' \
          '"https://www.googleapis.com/auth/servicecontrol",' \
          '"https://www.googleapis.com/auth/service.management.readonly"' \
          ',"https://www.googleapis.com/auth/trace.append" --num-nodes "3" ' \
          '--network "default" --enable-cloud-logging --enable-cloud-monitoring ' \
          '--subnetwork "default" --enable-legacy-authorization'