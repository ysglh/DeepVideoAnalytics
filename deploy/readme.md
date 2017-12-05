# Deep Video Analytics deployment configurations

- /dockerfiles contains Dockerfiles required for building containers
- /aws contains files used for launching DVA in a scalable Heroku + ECS/EC2 + S3 setup
- /gcp contains files used for launching DVA in a scalable GKE + GCS setup
- /single contains docker-compose files for non-GPU and GPU single machine deployments on Linode, AWS, GCP etc.
- /dev contains a docker-compose file to run postgres/rabbitmq locally for development