# Deep Video Analytics deployment configurations

#### Three deployment scenarios.

1. /cpu contains docker-compose files for non-GPU single machine deployments on Linode, AWS, GCP etc.

2. /gpu contains docker-compose files for GPU single machine deployments on AWS etc.

3. /gcp contains files used for launching DVA in a scalable GKE + GCS setup

#### Container images and development

- /dockerfiles contains Dockerfiles required for building containers

- /dev contains a docker-compose file to run Postgresql & RabbitMQ locally for development