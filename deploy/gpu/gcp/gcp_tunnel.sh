#/usr/bin/env bash
gcloud compute ssh $1 --zone us-west1-b -- -L 8600:localhost:8000 -L 8688:localhost:8888
