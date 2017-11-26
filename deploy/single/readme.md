# Single Machine deployment

The docker-compose files in this repo are intended for single instance deployment. 
E.g. A single EC2 or a GCP instance. For several type of workloads this is good enough.

- docker-compose-linode.yml : for single node deployments on VPS servers with port attacked to loopback interface and
                              used via SSH tunnel.

- docker-compose-gpu.yml : Same as above except when a GPU is available and NVidia docker compose is installled.                               

- docker-compose-linode-non-nfs.yml : used for testing now recommended non-NFS setup where a shared volume/fs is not available.

- docker-compose-gpu-low-memory.yml : same as GPU except indexers do not use GPU, lower GPU memory consumption. 
