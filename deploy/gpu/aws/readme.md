# Instructions for AWS

#### These instructions are tested with AWS P2 instances
#### We provide a public AMI in us-east-1 region with Docker, nvidia-docker2 and DVA GPU container image : ami-642f631e


- launch_aws.py : Launch on demand GPU instance on AWS

- packer_ami.json : Contains Packer script to automatically create AWS EC2 AMI using AWS Deep Learning AMI
                    in us-east-1 region.

- fabfile.py : Upon launch automatically connects and starts DVA containers. Also connects an SSH tunnel.
