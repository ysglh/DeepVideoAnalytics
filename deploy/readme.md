# Deep Video Analytics deployment configurations

#### Environment variables

- common.env contains environment variable used in both cpu and gpu/multi-gpu deployments.

#### Developing Deep Video Analytics

- /dev contains a docker-compose file which maps host /server directory (relative path), and can be used
       for interactively development and testing.

#### Three deployment scenarios.

1. /cpu contains docker-compose files for non-GPU single machine deployments on Linode, AWS, GCP etc.

2. /gpu contains docker-compose files for GPU single machine deployments on GCP & AWS etc.

3. /kube contains files used for launching DVA in a scalable GKE + GCS setup

#### Container images

- /dockerfiles contains Dockerfiles required for building containers