import logging, boto3, json, base64
from fabric.api import task, local, lcd, env

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/cloud.log',
                    filemode='a')

try:
    from config import KEY_FILE,AMI,IAM_ROLE,SecurityGroupId,EFS_DNS,KeyName,FLEET_ROLE,SecurityGroup,\
        CONFIG_BUCKET,ECS_AMI,ECS_ROLE,CLUSTER_NAME,ECS_GPU_AMI,GPU_CLUSTER_NAME
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
def launch(gpu_count=1,cpu_count=1,cpu_price=0.1,gpu_price=0.4):
    """
    Launch Spot fleet with instances using ECS AMI into an ECS cluster.
    The cluster can be then used to run task definitions.
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2spec_cpu = dict(ImageId=ECS_AMI,
                       KeyName=KeyName,
                       SecurityGroups=[{'GroupId': SecurityGroupId},],
                       InstanceType="c4.xlarge",
                       UserData=base64.b64encode(file('initdata/efs_ecs_bootstrap.txt').read().format(EFS_DNS,CLUSTER_NAME)),
                       WeightedCapacity=float(cpu_count),
                       Placement={
                           "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                       },
                       IamInstanceProfile=ECS_ROLE)
    ec2spec_gpu = dict(ImageId=ECS_GPU_AMI,
                       KeyName=KeyName,
                       SecurityGroups=[{'GroupId': SecurityGroupId},],
                       InstanceType="p2.xlarge",
                       UserData=base64.b64encode(file('initdata/efs_ecs_bootstrap_gpu.txt').read().format(EFS_DNS,GPU_CLUSTER_NAME)),
                       WeightedCapacity=float(gpu_count),
                       Placement={
                           "AvailabilityZone":"us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                       },
                       IamInstanceProfile=ECS_ROLE)
    for count,spec,price,itype in [(cpu_count,ec2spec_cpu,cpu_price,'CPU'),
                                   (gpu_count,ec2spec_gpu,gpu_price,'GPU')]:
        if count and int(count):
            launch_spec = [spec,]
            SpotFleetRequestConfig = dict(AllocationStrategy='lowestPrice',
                                          SpotPrice=str(price),
                                          TargetCapacity=int(count),
                                          IamFleetRole=FLEET_ROLE,
                                          InstanceInterruptionBehavior='stop',
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

@task
def heroku_migrate():
    """
    Migrate heroku postgres database
    """
    local('heroku run python manage.py migrate')


@task
def heroku_shell():
    """
    Launch heroku django shell for remote debug
    """
    local('heroku run python manage.py shell')


@task
def heroku_bash():
    """
    Launch heroku bash for remote debug
    """
    local('heroku run bash')


@task
def heroku_config():
    """
    View heroku config
    """
    local('heroku config')


@task
def heroku_psql():
    """
    Launch heroku database shell for remote debug
    """
    local('heroku pg:psql')


@task
def heroku_reset(bucket_name):
    """
    Reset heroku database and empty S3 bucket used for www.deepvideoanalytics.com
    """
    if raw_input("Are you sure type yes >>") == 'yes':
        local('heroku pg:reset DATABASE_URL')
        heroku_migrate()
        local('heroku run python manage.py createsuperuser')
        print "emptying bucket"
        local("aws s3 rm s3://{} --recursive --quiet".format(bucket_name))


@task
def heroku_super():
    """
    Create superuser for heroku
    """
    local('heroku run python manage.py createsuperuser')


@task
def heroku_setup():
    """
    Setup heroku by adding custom buildpack for private repo and disabling collect static
    """
    local('heroku buildpacks:add https://github.com/AKSHAYUBHAT/heroku-buildpack-run.git')
    local('heroku config:set DISABLE_COLLECTSTATIC=1')


@task
def sync_static(bucket_name='dvastatic'):
    """
    Sync static folder with AWS S3 bucket
    :param bucket_name:
    """
    with lcd('../'):
        local("python manage.py collectstatic")
        with lcd('dva'):
            local('aws s3 sync staticfiles/ s3://{}/'.format(bucket_name))


@task
def enable_media_bucket_static_hosting(bucket_name, allow_videos=False):
    """
    Enable static hosting for given bucket name
    Note that the bucket / media becomes publicly viewable.
    An alternative is using presigned url but it will require a django filter
    https://stackoverflow.com/questions/33549254/how-to-generate-url-from-boto3-in-amazon-web-services
    :param bucket_name:
    :param allow_videos: set True if you wish to serve videos from S3 (costly!)
    """
    s3 = boto3.client('s3')
    cors_configuration = {
        'CORSRules': [{
            'AllowedHeaders': ['*'],
            'AllowedMethods': ['GET'],
            'AllowedOrigins': ['*'],
            'ExposeHeaders': ['GET'],
            'MaxAgeSeconds': 3000
        }]
    }
    s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_configuration)
    bucket_policy = {
        'Version': '2012-10-17',
        'Statement': [{
            'Sid': 'AddPerm',
            'Effect': 'Allow',
            'Principal': '*',
            'Action': ['s3:GetObject'],
            'Resource': "arn:aws:s3:::%s/*.jpg" % bucket_name
        },
            {
                'Sid': 'AddPerm',
                'Effect': 'Allow',
                'Principal': '*',
                'Action': ['s3:GetObject'],
                'Resource': "arn:aws:s3:::%s/*.png" % bucket_name
            }]
    }
    if allow_videos:
        bucket_policy['Statement'].append({
            'Sid': 'AddPerm',
            'Effect': 'Allow',
            'Principal': '*',
            'Action': ['s3:GetObject'],
            'Resource': "arn:aws:s3:::%s/*.mp4" % bucket_name
        })
    bucket_policy = json.dumps(bucket_policy)
    s3.put_bucket_policy(Bucket=bucket_name, Policy=bucket_policy)
    website_configuration = {'ErrorDocument': {'Key': 'error.html'}, 'IndexDocument': {'Suffix': 'index.html'}, }
    s3.put_bucket_website(Bucket=bucket_name, WebsiteConfiguration=website_configuration)


@task
def make_requester_pays(bucket_name):
    """
    Convert AWS S3 bucket into requester pays bucket
    :param bucket_name:
    """
    s3 = boto3.resource('s3')
    bucket_request_payment = s3.BucketRequestPayment(bucket_name)
    _ = bucket_request_payment.put(RequestPaymentConfiguration={'Payer': 'Requester'})
    bucket_policy = s3.BucketPolicy(bucket_name)
    policy = {
        "Id": "Policy1493037034955",
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Stmt1493036947566",
                "Action": [
                    "s3:ListBucket"
                ],
                "Effect": "Allow",
                "Resource": "arn:aws:s3:::{}".format(bucket_name),
                "Principal": "*"
            },
            {
                "Sid": "Stmt1493037029723",
                "Action": [
                    "s3:GetObject"
                ],
                "Effect": "Allow",
                "Resource": "arn:aws:s3:::{}/*".format(bucket_name),
                "Principal": {
                    "AWS": [
                        "*"
                    ]
                }
            }
        ]}
    _ = bucket_policy.put(Policy=json.dumps(policy))


@task
def get_envs():
    local("")