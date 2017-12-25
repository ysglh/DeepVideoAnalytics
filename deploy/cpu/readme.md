# Deployment on a single machine without GPU

E.g. A desktop, linode, digital ocean droplet, aws ec2 instance or a gcp cloud vm.

- docker-compose.yml : for single node deployments.
                              
- docker-compose-non-nfs.yml : for testing non-NFS setup where S3 or GCS is used instead of a shared volume.

- shell.sh : For bash into a specific running container e.g. `./shell.sh inception`

- webserver_logs.sh : Get uwsgi logs from webserver

#### Security warning

When deploying/running on remote Ubuntu machines on VPS services such as Linode etc. please be aware of the
[Docker/UFW firewall issues](https://askubuntu.com/questions/652556/uncomplicated-firewall-ufw-is-not-blocking-anything-when-using-docker).
Docker bypasses UFW firewall and opens the port 8000 to internet.
You can change the behavior by using a loopback interface (127.0.0.1:8000:80) and then forwarding the
local port (8000) over SSH tunnel.