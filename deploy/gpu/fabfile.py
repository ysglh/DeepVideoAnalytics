import logging, boto3, json, base64, glob
from fabric.api import task, local, lcd, env

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../../logs/cloud.log',
                    filemode='a')

try:
    from config import KEY_FILE,AMI,IAM_ROLE,SecurityGroupId,KeyName,FLEET_ROLE,SecurityGroup,\
        CONFIG_BUCKET,ECS_AMI,ECS_ROLE,CLUSTER_NAME,ECS_GPU_AMI,GPU_CLUSTER_NAME
except ImportError:
    raise ImportError,"Please create config.py with KEY_FILE,AMI,IAM_ROLE,SecurityGroupId,KeyName"

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
def launch_spot():
    """
    Launch Spot fleet with instances using ECS AMI into an ECS cluster.
    The cluster can be then used to run task definitions.
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2spec_gpu = dict(ImageId=ECS_GPU_AMI,
                       KeyName=KeyName,
                       SecurityGroups=[{'GroupId': SecurityGroupId},],
                       InstanceType="p2.xlarge",
                       WeightedCapacity=float(1.0),
                       Placement={
                           "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                       },
                       IamInstanceProfile=ECS_ROLE)
    count,spec,price,itype = 1,ec2spec_gpu,1.0,'GPU'
    launch_spec = [spec,]
    SpotFleetRequestConfig = dict(AllocationStrategy='lowestPrice', SpotPrice=str(price), TargetCapacity=int(count),
                                  IamFleetRole=FLEET_ROLE, InstanceInterruptionBehavior='stop',
                                  LaunchSpecifications=launch_spec)
    output = ec2.request_spot_fleet(DryRun=False, SpotFleetRequestConfig=SpotFleetRequestConfig)
    fleet_request_id = output[u'SpotFleetRequestId']
    print "Lauched fleet request with id {} for {} {} instances at {} price".format(fleet_request_id,count,itype,price)




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