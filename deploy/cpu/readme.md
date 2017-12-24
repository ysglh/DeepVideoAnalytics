# Deployment on a single machine without GPU

The docker-compose files in this repo are intended for single instance non-GPU deployment.

E.g. A single Linode, Digital Ocean, EC2 or a GCP instance.

- docker-compose-linode.yml : for single node deployments on VPS servers with port attacked to loopback interface and
                              used via SSH tunnel.
                              
- docker-compose-linode-non-nfs.yml : used for testing non-NFS setup where S3 or GCS is used instead of a shared volume/fs.

#### Security warning

When deploying/running on remote Ubuntu machines on VPS services such as Linode etc. please be aware of the
[Docker/UFW firewall issues](https://askubuntu.com/questions/652556/uncomplicated-firewall-ufw-is-not-blocking-anything-when-using-docker). Docker bypasses UFW firewall and opens the port 8000 to internet.
You can change the behavior by using a loopback interface (127.0.0.1:8000:80) and then forwarding the
port (8000) over SSH tunnel.