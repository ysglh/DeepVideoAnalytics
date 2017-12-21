# Deployment on a single machine without GPU

The docker-compose files in this repo are intended for single instance non-GPU deployment. 
E.g. A single Linode, Digital Ocean, EC2 or a GCP instance.

- docker-compose-linode.yml : for single node deployments on VPS servers with port attacked to loopback interface and
                              used via SSH tunnel.
                              
- docker-compose-linode-non-nfs.yml : used for testing now recommended non-NFS setup where a shared volume/fs is not available.