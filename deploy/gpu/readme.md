# Single machine with GPU deployment

The docker-compose files in this repo are intended for single instance GPU deployment. 
E.g. A single EC2 or a GCP instance. For several type of workloads this is good enough.

- docker-compose-gpu.yml : Same as above except when a GPU is available and NVidia docker compose is installled.
                               
- docker-compose-gpu-low-memory.yml : Same as GPU except indexers do not use GPU for lower GPU memory consumption.

- fix_docker_compose.py : make nvidia-docker default runtime.

- install_docker.sh :  install compatible docker version, make sure you log out and log in.

- install_nvidia_docker.sh install nvidia docker and make it default runtime.

- launch.py : Launch on demand GPU instance on AWS

- launch_spot.py : Launch spot GPU instance on AWS