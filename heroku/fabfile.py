import logging, boto3, json
from fabric.api import task, local, lcd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='../logs/heroku.log',
                    filemode='a')


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
    with lcd('../dva'):
        local("python manage.py collectstatic")
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
