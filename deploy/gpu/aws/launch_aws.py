#!/usr/bin/env python
import logging, boto3, subprocess

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../../logs/cloud.log',
                    filemode='a')
from config import AMI,KeyName,SecurityGroupName,IAM_ROLE,env_user,key_filename

if __name__ == '__main__':
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    instances = ec2r.create_instances(DryRun=False, ImageId=AMI, KeyName=KeyName, MinCount=1, MaxCount=1,
                                      SecurityGroups=[SecurityGroupName, ], InstanceType="p2.xlarge",
                                      Monitoring={'Enabled': True, },BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                                                                           "Ebs" : { "VolumeSize" : 200 }}],
                                      IamInstanceProfile=IAM_ROLE)
    for instance in instances:
        instance.wait_until_running()
        instance.reload()
        print(instance.id, instance.instance_type)
        logging.info("instance allocated")
        with open('host','w') as h:
            h.write(instance.public_ip_address)
        fh = open("connect.sh", 'w')
        fh.write(
            "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -L 8688:localhost:8888 -i ' + key_filename + " " + env_user + "@" +
            instance.public_ip_address + "\n")
        fh.close()
    subprocess.call(['fab','deploy'])