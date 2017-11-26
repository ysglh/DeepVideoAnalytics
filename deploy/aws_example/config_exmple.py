AMI = ''  # AMI
KeyName = ''  # REPLACE WITH YOUR OWN
SecurityGroupId = ''  # REPLACE WITH YOUR OWN e.g. sg-asd*
SecurityGroup = ''  # REPLACE WITH YOUR OWN
IAM_ROLE = {'Arn': '', }  # REPLACE WITH YOUR OWN (Provide S3 access to allow sync) E.g. arn:aws:iam::23123123:instance-profile/role_that_gives_s3_access
KEY_FILE = ""  # REPLACE WITH YOUR OWN Path to private key
# EFS DNS no longer needed
FLEET_ROLE = "" # e.g. arn:aws:iam::123123123123123:role/aws-ec2-spot-fleet-role
CONFIG_BUCKET = "" # AWS S3 private bucket containing heroku.env credentials
ECS_AMI = "ami-9eb4b1e5" # us-east-1 AMI Replace this with latest ECS AMI if you are launching in a different region
ECS_GPU_AMI = "ami-b1c106cb" # Create your own modified AWS Deep Learning AMI with NVidia docker and ECS agent installed
ECS_ROLE = {'Arn':''} # REPLACE WITH YOUR OWN (Attach both S3 AND ECSforEC2) E.g. arn:aws:iam::23123123:instance-profile/role_that_gives_s3_access
CLUSTER_NAME = ''
GPU_CLUSTER_NAME = '' # have a seperate cluster of GPU instances to prevent underutilization
MEDIA_BUCKET = '' # media bucket name