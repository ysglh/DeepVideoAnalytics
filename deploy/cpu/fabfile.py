from fabric.api import task, local


@task
def shell(container_name="dva-server"):
    local('docker exec -u="root" -it {} bash'.format(container_name))


@task
def aws_configure():
    local('docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli && aws configure"')


@task
def copy_aws_creds_to_docker():
    local('docker cp ~/.aws dva-server:/root/.aws')


@task
def docker_superu():
    local('docker exec -u="root" -it dva-server python manage.py createsuperuser')


@task
def notebook():
    local('docker exec -u="root" -it dva-server bash -c "pip install --upgrade jupyter"')
    local('docker exec -u="root" -it dva-server bash -c "jupyter notebook --allow-root"')