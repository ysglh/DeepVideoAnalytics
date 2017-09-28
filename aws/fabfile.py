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

USER_DATA = """
#cloud-boothook
#!/bin/sh
set -x
sudo mkdir /efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 {}:/ /efs
export SECRET_KEY=$'{}'
export DATABASE_URL=$'{}'
export BROKER_URL=$'{}'
export MEDIA_BUCKET=$'{}'
sudo chown ubuntu:ubuntu /efs/
cd /efs && mkdir media
service docker restart
docker volume create --opt type=none --opt device=/efs/media --opt o=bind dvadata
cd /home/ubuntu/DeepVideoAnalytics && git pull
docker rmi akshayubhat/dva-auto:gpu
docker rmi akshayubhat/dva-auto:caffe
sudo pip install --upgrade nvidia-docker-compose
""".format(EFS_DNS,SECRET_KEY,DATABASE_URL,BROKER_URL,MEDIA_BUCKET)

def get_status(ec2, spot_request_id):
    """
    Get status of EC2 spot request
    :param ec2:
    :param spot_request_id:
    :return:
    """
    current = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id, ])
    instance_id = current[u'SpotInstanceRequests'][0][u'InstanceId'] if u'InstanceId' in \
                                                                        current[u'SpotInstanceRequests'][0] else None
    return instance_id


@task
def launch(container=False):
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    compose_file = """
    cd /home/ubuntu/DeepVideoAnalytics/docker && nvidia-docker-compose -f {} up -d > launch.log 2>error.log &
    """.format("custom/docker-compose-gpu.yml" if container else "custom/docker-compose-worker-gpu.yml")
    user_data = USER_DATA + compose_file
    ec2spec = dict(ImageId=AMI,
                   KeyName=KeyName,
                   SecurityGroups=[{'GroupId': SecurityGroupId},],
                   InstanceType="p2.xlarge",
                   Monitoring={'Enabled': True,},
                   UserData=base64.b64encode(user_data),
                   Placement={
                       "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                   },
                   IamInstanceProfile=IAM_ROLE)
    SpotFleetRequestConfig = dict(AllocationStrategy='lowestPrice',
                                  SpotPrice = "0.9",
                                  TargetCapacity = 1,
                                  IamFleetRole = FLEET_ROLE,
                                  LaunchSpecifications = [ec2spec,])
    output = ec2.request_spot_fleet(DryRun=False,SpotFleetRequestConfig=SpotFleetRequestConfig)
    fleet_request_id = output[u'SpotFleetRequestId']
    logging.info("instance requested")
    time.sleep(30)
    waiter = ec2.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[spot_request_id, ])
    instance_id = get_status(ec2, spot_request_id)
    while instance_id is None:
        time.sleep(30)
        instance_id = get_status(ec2, spot_request_id)
    instance = ec2r.Instance(instance_id)
    with open("host", 'w') as out:
        out.write(instance.public_ip_address)
    logging.info("instance allocated")
    time.sleep(10)  # wait while the instance starts
    env.hosts = [instance.public_ip_address, ]
    fh = open("connect.sh", 'w')
    fh.write(
        "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -i ' + env.key_filename + " " + env.user + "@" +
        instance.public_ip_address + "\n")
    fh.close()


@task
def launch_on_demand():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
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


@task
def launch_cpu():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    ec2spec = dict(ImageId=AMI,
                   KeyName=KeyName,
                   SecurityGroupIds=[SecurityGroupId, ],
                   InstanceType="c4.2xlarge",
                   Monitoring={'Enabled': True, },
                   IamInstanceProfile=IAM_ROLE)
    output = ec2.request_spot_instances(DryRun=False,
                                        SpotPrice="0.4",
                                        InstanceCount=1,
                                        LaunchSpecification=ec2spec)
    spot_request_id = output[u'SpotInstanceRequests'][0][u'SpotInstanceRequestId']
    logging.info("instance requested")
    time.sleep(30)
    waiter = ec2.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[spot_request_id, ])
    instance_id = get_status(ec2, spot_request_id)
    while instance_id is None:
        time.sleep(30)
        instance_id = get_status(ec2, spot_request_id)
    instance = ec2r.Instance(instance_id)
    with open("host", 'w') as out:
        out.write(instance.public_ip_address)
    logging.info("instance allocated")
    time.sleep(10)  # wait while the instance starts
    env.hosts = [instance.public_ip_address, ]
    fh = open("connect.sh", 'w')
    fh.write(
        "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -i ' + env.key_filename + " " + env.user + "@" +
        env.hosts[0] + "\n")
    fh.close()
    local("fab deploy_cpu")


@task
def launch_extractor_cpu():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    To use this please change the keyname, security group and IAM roles at the top
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    ec2spec = dict(ImageId=AMI,
                   KeyName=KeyName,
                   SecurityGroupIds=[SecurityGroupId, ],
                   InstanceType="c4.2xlarge",
                   Monitoring={'Enabled': True, },
                   IamInstanceProfile=IAM_ROLE)
    output = ec2.request_spot_instances(DryRun=False,
                                        SpotPrice="0.4",
                                        InstanceCount=1,
                                        LaunchSpecification=ec2spec)
    spot_request_id = output[u'SpotInstanceRequests'][0][u'SpotInstanceRequestId']
    logging.info("instance requested")
    time.sleep(30)
    waiter = ec2.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[spot_request_id, ])
    instance_id = get_status(ec2, spot_request_id)
    while instance_id is None:
        time.sleep(30)
        instance_id = get_status(ec2, spot_request_id)
    instance = ec2r.Instance(instance_id)
    with open("host", 'w') as out:
        out.write(instance.public_ip_address)
    logging.info("instance allocated")
    time.sleep(10)  # wait while the instance starts
    env.hosts = [instance.public_ip_address, ]
    fh = open("connect.sh", 'w')
    fh.write(
        "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -i ' + env.key_filename + " " + env.user + "@" +
        env.hosts[0] + "\n")
    fh.close()
    local("fab deploy_cpu:custom/docker-compose-extractor-worker.yml")


@task
def deploy(compose_file="custom/docker-compose-worker-gpu.yml", dns=EFS_DNS):
    """
    deploys code on hostname
    :return:
    """
    import webbrowser
    time.sleep(120)
    for attempt in range(3):
        try:
            run('ls')  # just run some command that has no effect to ensure you dont get timed out
            break  # break if you succeed
        except:
            time.sleep(120)
            pass
    mount_efs(dns)
    run('docker rmi akshayubhat/dva-auto:gpu')
    run('docker rmi akshayubhat/dva-auto:caffe')
    with cd('DeepVideoAnalytics'):
        run('git pull')
        with cd("docker"):
            run('aws s3 cp s3://aub3config/heroku.env heroku.env')
            run('source heroku.env && nvidia-docker-compose -f {} up -d'.format(compose_file))


@task
def deploy_container(compose_file="custom/docker-compose-gpu.yml"):
    """
    deploys code on hostname
    :return:
    """
    import webbrowser
    time.sleep(120)
    for attempt in range(3):
        try:
            run('ls')  # just run some command that has no effect to ensure you dont get timed out
            break  # break if you succeed
        except:
            time.sleep(120)
            pass
    with cd('DeepVideoAnalytics'):
        run('git pull')
        with cd("docker"):
            run('nvidia-docker-compose -f {} up -d'.format(compose_file))


@task
def deploy_cpu(compose_file="custom/docker-compose-worker-cpu.yml", dns=EFS_DNS):
    """
    deploys code on hostname
    :return:
    """
    import webbrowser
    for attempt in range(3):
        try:
            run('ls')  # just run some command that has no effect to ensure you dont get timed out
            break  # break if you succeed
        except:
            time.sleep(120)
            pass
    mount_efs(dns)
    with cd('DeepVideoAnalytics'):
        with cd("docker"):
            run('git pull')
            run('aws s3 cp s3://aub3config/heroku.env heroku.env')
            run('git pull && docker pull akshayubhat/dva-auto:latest')
            run('source heroku.env && docker-compose -f {} up -d'.format(compose_file))


@task
def mount_efs(dns):
    # sudo('apt-get install -y nfs-common')
    sudo('mkdir /efs')
    sudo('mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 {}:/ /efs'.format(dns))
    sudo('chown ubuntu:ubuntu /efs/')
    with cd('/efs'):
        try:
            run('mkdir media')
        except:
            pass
    sudo('service docker restart')
    run('docker volume create --opt type=none --opt device=/efs/media --opt o=bind dvadata')


# @task
# def create_efs():
    # client = boto3.client('efs')
    # if os.path.isfile('efs.json'):
    #     raise ValueError, "efs.json already exists please delete it"
    # else:
    #     response = client.create_file_system(CreationToken=str(uuid.uuid1()),
    #                                          PerformanceMode='generalPurpose',
    #                                          Encrypted=False)
    #     print response
    #     with open('efs.json', 'w') as out:
    #         out.write(json.dumps(response))
    # response = client.create_mount_target(
    #     FileSystemId='fs-01234567',
    #     SubnetId='subnet-1234abcd',
    # )
    # client = boto3.client('efs')
    # if not os.path.isfile('efs.json'):
    #     raise ValueError, "efs.json does not exists"
    # else:
    #     efs = json.loads(open('efs.json').read())
    #     print client.describe_file_system(FileSystemId=efs['FileSystemId'])