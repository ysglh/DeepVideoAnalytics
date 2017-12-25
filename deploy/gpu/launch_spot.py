import boto3

ECS_GPU_AMI = ""
KeyName = ""
SecurityGroupId = ""
ECS_ROLE = ""
FLEET_ROLE = ""


if __name__ == '__main__':
    ec2 = boto3.client('ec2')
    ec2spec_gpu = dict(ImageId=ECS_GPU_AMI,
                       KeyName=KeyName,
                       SecurityGroups=[{'GroupId': SecurityGroupId}, ],
                       InstanceType="p2.xlarge",
                       WeightedCapacity=float(1.0),
                       Placement={
                           "AvailabilityZone": "us-east-1a,us-east-1b,us-east-1c,us-east-1d,us-east-1e,us-east-1f"
                       },
                       IamInstanceProfile=ECS_ROLE)
    count, spec, price, itype = 1, ec2spec_gpu, 1.0, 'GPU'
    launch_spec = [spec, ]
    SpotFleetRequestConfig = dict(AllocationStrategy='lowestPrice', SpotPrice=str(price), TargetCapacity=int(count),
                                  IamFleetRole=FLEET_ROLE, InstanceInterruptionBehavior='stop',
                                  LaunchSpecifications=launch_spec)
    output = ec2.request_spot_fleet(DryRun=False, SpotFleetRequestConfig=SpotFleetRequestConfig)
    fleet_request_id = output[u'SpotFleetRequestId']
    print "Lauched fleet request with id {} for {} {} instances at {} price".format(fleet_request_id, count, itype, price)
