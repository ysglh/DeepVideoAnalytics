AMI = ''  # AMI
KeyName = ''  # REPLACE WITH YOUR OWN
SecurityGroupId = ''  # REPLACE WITH YOUR OWN e.g. sg-asd*
SecurityGroup = ''  # REPLACE WITH YOUR OWN
IAM_ROLE = {'Arn': '', }  # REPLACE WITH YOUR OWN (Provide S3 access to allow sync) E.g. arn:aws:iam::23123123:instance-profile/role_that_gives_s3_access
KEY_FILE = ""  # REPLACE WITH YOUR OWN Path to private key
EFS_DNS = "" #  e.g. fs-asdasdasdasd.efs.us-east-1.amazonaws.com
FLEET_ROLE = "" # e.g. arn:aws:iam::123123123123123:role/aws-ec2-spot-fleet-role
CONFIG_BUCKET = "" # AWS S3 private bucket containing heroku.env credentials
ECS_AMI = "ami-9eb4b1e5"

