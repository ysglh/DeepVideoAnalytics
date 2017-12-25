import logging, boto3

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../../logs/cloud.log',
                    filemode='a')
env_user = "ubuntu"
key_filename = ""
AMI = ""
KeyName = ""
SecurityGroupId = ""
IAM_ROLE = ""

if __name__ == '__main__':
    ec2 = boto3.client('ec2')
    ec2r = boto3.resource('ec2')
    instances = ec2r.create_instances(DryRun=False, ImageId=AMI, KeyName=KeyName, MinCount=1, MaxCount=1,
                                      SecurityGroups=[SecurityGroupId, ], InstanceType="p2.xlarge",
                                      Monitoring={'Enabled': True, }, IamInstanceProfile=IAM_ROLE)
    for instance in instances:
        instance.wait_until_running()
        instance.reload()
        print(instance.id, instance.instance_type)
        logging.info("instance allocated")
        with open("host", 'w') as out:
            out.write(instance.public_ip_address)
        fh = open("connect.sh", 'w')
        fh.write(
            "#!/bin/bash\n" + 'autossh -M 0 -o "ServerAliveInterval 30" -o "ServerAliveCountMax 3" -L 8600:localhost:8000 -i ' + KeyName + " " + env_user + "@" +
            instance.public_ip_address + "\n")
        fh.close()