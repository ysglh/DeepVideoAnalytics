# Single machine with GPU deployment

```bash
# And nvidia-docker is default runtime https://github.com/NVIDIA/nvidia-docker/issues/568
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics
cd DeepVideoAnalytics/deploy/gpu
docker-compose -f docker-compose-gpu.yml up
# Above command will automatically pull container images from docker-hub
# You can visit Web UI on localhost:8600
```

The docker-compose files in this repo are intended for single instance GPU deployment. 
E.g. A single EC2 or a GCP instance. For several type of workloads this is good enough.

- docker-compose-gpu.yml : Same as above except when a GPU is available and NVidia docker compose is installled.
                               
- docker-compose-gpu-low-memory.yml : same as GPU except indexers do not use GPU for lower GPU memory consumption. 
