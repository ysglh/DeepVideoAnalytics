import os,logging,time,boto3
from fabric.api import task,local,run,put,get,lcd,cd,sudo,env,puts
import django,sys
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/fab.log',
                    filemode='a')
AMI = 'ami-b3cc1fa5'
KeyName = 'C:\\fakepath\\cs5356'
SecurityGroupId = 'sg-06a6b562'
IAM_ROLE = {'Arn': 'arn:aws:iam::248089713624:instance-profile/chdeploy',}
env.user = "ubuntu"
try:
    ec2_HOST = file("host").read().strip()
except:
    ec2_HOST = ""
    pass
env.key_filename = "~/.ssh/cs5356"




def get_status(ec2,spot_request_id):
    current = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[spot_request_id,])
    instance_id = current[u'SpotInstanceRequests'][0][u'InstanceId'] if u'InstanceId' in current[u'SpotInstanceRequests'][0] else None
    return instance_id


@task
def launch_spot():
    """
    A helper script to launch a spot P2 instance running Deep Video Analytics
    :return:
    """
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    ec2spec = dict(ImageId=AMI,
                   KeyName = KeyName,
                   SecurityGroupIds = [SecurityGroupId, ],
                   InstanceType = "p2.xlarge",
                   Monitoring = {'Enabled': True,},
                   IamInstanceProfile = IAM_ROLE)
    output = ec2.request_spot_instances(DryRun=False,
                                        SpotPrice="0.9",
                                        InstanceCount=1,
                                        LaunchSpecification = ec2spec)
    spot_request_id = output[u'SpotInstanceRequests'][0][u'SpotInstanceRequestId']
    time.sleep(30)
    waiter = ec2.get_waiter('spot_instance_request_fulfilled')
    waiter.wait(SpotInstanceRequestIds=[spot_request_id,])
    instance_id = get_status(ec2, spot_request_id)
    while instance_id is None:
        time.sleep(30)
        instance_id = get_status(ec2,spot_request_id)
    instance = ec2r.Instance(instance_id)
    with open("host",'w') as out:
        out.write(instance.public_ip_address)
    time.sleep(12) # wait while the instance starts
    env.hosts = [instance.public_ip_address,]
    fh = open("connect.sh", 'w')
    fh.write("#!/bin/bash\n" + "ssh -i " + env.key_filename + " " + env.user + "@" + env.hosts[0] + "\n")
    fh.close()
    local("fab deploy_ec2")

@task
def deploy_ec2():
    import webbrowser
    run('cd deepvideoanalytics && git pull && cd docker_GPU && ./rebuild.sh && nvidia-docker-compose up -d')
    webbrowser.open('{}:8000'.format(env.hosts[0]))

@task
def shell():
    local('python manage.py shell')



@task
def local_static():
    local('python manage.py collectstatic')


@task
def migrate():
    local('python manage.py makemigrations')
    local('python manage.py migrate')


@task
def worker(queue_name,conc=1):
    conc = int(conc)
    command = 'celery -A dca worker -l info -c {} -Q {} -n {}.%h -f logs/celery.log'.format(conc,queue_name,queue_name)
    if sys.platform != 'darwin':
        command = "source ~/.profile && "+command
    local(command=command)


@task
def server():
    local("python manage.py runserver")


@task
def start_server_container(test=False):
    local('sleep 60')
    migrate()
    local('python startq.py extractor &')
    local('python startq.py indexer &')
    local('python startq.py detector &')
    if test:
        ci()
    local('python manage.py runserver 0.0.0.0:8000')


@task
def start_server_container_gpu(test=False):
    local('sleep 60')
    migrate()
    local('chmod 0777 -R /tmp')
    local("mv docker_GPU/configs/nginx.conf /etc/nginx/")
    local("mv docker_GPU/configs/nginx-app.conf /etc/nginx/sites-available/default")
    local("mv docker_GPU/configs/supervisor-app.conf /etc/supervisor/conf.d/")
    local("python manage.py collectstatic --no-input")
    local("chmod 0777 -R dva/staticfiles/")
    local("chmod 0777 -R dva/media/")
    local('python startq.py extractor 2 &') # on GPU machines use more extractors
    local('python startq.py indexer &')
    local('python startq.py detector &')
    if test:
        ci()
    local('supervisord -n')


@task
def clean():
    for qname in ['qextract','qindexer','qdetector']:
        try:
            local('rabbitmqadmin purge queue name={}'.format(qname))
        except:
            logging.warning("coudnt clear queue {}".format(qname))
    local('python manage.py makemigrations')
    local('python manage.py flush --no-input')
    migrate()
    local("rm logs/*.log")
    local("rm -rf ~/media/*")
    local("mkdir ~/media/queries")
    try:
        local("ps auxww | grep 'celery -A dva worker' | awk '{print $2}' | xargs kill -9")
    except:
        pass


@task
def ci():
    with lcd("tests"):
        local('wget https://www.dropbox.com/s/xtpkb18i6hn39ht/Goldeneye.mp4')
        local('wget https://www.dropbox.com/s/cjo9b68poqk7gy2/TomorrowNeverDies.mp4')
        local('wget https://www.dropbox.com/s/xwbk5g1qit5s9em/WorldIsNotEnough.mp4')
        local('wget https://www.dropbox.com/s/misaejsbz6722pd/test.png')
    local('python run_test.py')


@task
def quick_test(detection=False):
    clean()
    create_super()
    local('python run_test.py')
    local('python startq.py extractor &')
    local('python startq.py indexer &')
    if detection:
        local('python startq.py detector &')

@task
def test_backup():
    local('python backup.py')
    clean()
    local('python restore.py')


@task
def create_super():
    local('echo "from django.contrib.auth.models import User; User.objects.create_superuser(\'akshay\', \'akshay@test.com\', \'super\')" | python manage.py shell')