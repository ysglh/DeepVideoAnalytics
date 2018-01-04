# Deployment on a single machine without GPU

E.g. A desktop, linode, digital ocean droplet, aws ec2 instance or a gcp cloud vm.

- docker-compose.yml : for single node deployments.
                              
- docker-compose-non-nfs.yml : for testing non-NFS setup where S3 or GCS is used instead of a shared volume.

- shell.sh : For bash into a specific running container e.g. `./shell.sh inception`

- webserver_logs.sh : Get uwsgi logs from webserver

#### The docker-compose files use loopback interface (127.0.0.1:8000:80), we recommend forwarding the host OS port (8000)
over SSH tunnel when using cloud providers or VPS services such as Linode.