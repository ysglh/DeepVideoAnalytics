# Single Machine deployment

The docker-compose files in this repo are intended for single instance GPU deployment. 
E.g. A single EC2 or a GCP instance. For several type of workloads this is good enough.

- docker-compose-gpu.yml : Same as above except when a GPU is available and NVidia docker compose is installled.
                               
- docker-compose-gpu-low-memory.yml : same as GPU except indexers do not use GPU for lower GPU memory consumption. 
