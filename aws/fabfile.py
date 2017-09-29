import os, logging, time, boto3, glob, subprocess, calendar, sys, uuid, json, base64
from fabric.api import task, local, run, put, get, lcd, cd, sudo, env, puts

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', filename='../logs/aws.log', filemode='a')

try:
    from config import KEY_FILE,AMI,IAM_ROLE,SecurityGroupId,EFS_DNS,KeyName,\
        SECRET_KEY,DATABASE_URL,BROKER_URL,MEDIA_BUCKET,FLEET_ROLE,SecurityGroup
except ImportError:
    raise ImportError,"Please create config.py with KEY_FILE,AMI,IAM_ROLE,SecurityGroupId,EFS_DNS,KeyName"

env.user = "ubuntu"  # DONT CHANGE

try:
    ec2_HOST = file("host").read().strip()
    env.hosts = [ec2_HOST, ]
except:
    ec2_HOST = ""
    logging.warning("No host file available assuming that the instance is not launched")
    pass

env.key_filename = KEY_FILE


@task
def launch(gpu_count=1,cpu_count=0):
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    :return:
    """
    ec2 = boto3.client('ec2')
    user_data_gpu = file('userdata/worker_gpu.sh').read().format(EFS_DNS,"custom/docker-compose-worker-gpu.yml")
    ec2spec_gpu = dict(ImageId=AMI,
                   KeyName=KeyName,
                   SecurityGroups=[{'GroupId': SecurityGroupId},],
                   InstanceType="p2.xlarge",
                   Monitoring={'Enabled': True,},
                   UserData=base64.b64encode(user_data_gpu),
                   WeightedCapacity=float(gpu_count),
                   Placement={
                       "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                   },
                   IamInstanceProfile=IAM_ROLE)
    user_data_cpu = file('userdata/worker_cpu.sh').read().format(EFS_DNS,"custom/docker-compose-worker-cpu.yml")
    ec2spec_cpu = dict(ImageId=AMI,
                   KeyName=KeyName,
                   SecurityGroups=[{'GroupId': SecurityGroupId},],
                   InstanceType="c4.xlarge",
                   Monitoring={'Enabled': True,},
                   UserData=base64.b64encode(user_data_cpu),
                   WeightedCapacity=float(cpu_count),
                   Placement={
                       "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                   },
                   IamInstanceProfile=IAM_ROLE)
    launch_spec = []
    if cpu_count and int(cpu_count):
        launch_spec.append(ec2spec_cpu)
    if gpu_count and int(gpu_count):
        launch_spec.append(ec2spec_gpu)
    SpotFleetRequestConfig = dict(AllocationStrategy='lowestPrice',
                                  SpotPrice = "0.9",
                                  TargetCapacity = int(cpu_count)+int(gpu_count),
                                  IamFleetRole = FLEET_ROLE,
                                  LaunchSpecifications = launch_spec)
    output = ec2.request_spot_fleet(DryRun=False,SpotFleetRequestConfig=SpotFleetRequestConfig)
    fleet_request_id = output[u'SpotFleetRequestId']
    print fleet_request_id


@task
def launch_on_demand():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    # apt-get install -y nfs-common
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    instances = ec2r.create_instances(DryRun=False, ImageId=AMI,
                                      KeyName=KeyName, MinCount=1, MaxCount=1,
                                      SecurityGroups=[SecurityGroupId, ],
                                      InstanceType="p2.xlarge",
                                      Monitoring={'Enabled': True, },
                                      IamInstanceProfile=IAM_ROLE)
    for instance in instances:
        instance.wait_until_running()
        instance.reload()
        print(instance.id, instance.instance_type)
        logging.info("instance allocated")
        with open("host", 'w') as out:
            out.write(instance.public_ip_address)
        env.hosts = [instance.public_ip_address, ]
        fh = open("connect.sh", 'w')
        fh.write(
            "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -i ' + env.key_filename + " " + env.user + "@" +
            env.hosts[0] + "\n")
        fh.close()



