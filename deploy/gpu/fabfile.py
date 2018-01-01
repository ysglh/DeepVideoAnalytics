import logging, time
from fabric.api import task, local, run, cd, env

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M', filename='../logs/aws.log', filemode='a')

from config import key_filename

env.user = "ubuntu"  # DONT CHANGE

try:
    ec2_HOST = file("host").read().strip()
    env.hosts = [ec2_HOST, ]
except:
    raise ValueError("No host file available assuming that the instance is not launched")


env.key_filename = key_filename


@task
def deploy(compose_file="docker-compose-gpu.yml"):
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
    with cd('DeepVideoAnalytics/deploy/gpu'):
        run('git pull')
        run('docker-compose -f {} up -d'.format(compose_file))
    time.sleep(120)
    webbrowser.open("http://localhost:8600")
    local("./connect.sh")