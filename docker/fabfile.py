from fabric.api import task, local


@task
def shell():
    local('docker exec -u="root" -it dva-server bash')


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
def setup_nginx_auth():
    try:
        local("rm .htpasswd")
    except:
        pass
    local("echo -n 'dvauser:' >> .htpasswd")
    local("openssl passwd -apr1 >> .htpasswd")
    local("cp .htpasswd dva-server:/etc/nginx/.htpasswd")
    local('docker exec -u="root" -it dva-server bash -c "supervisorctl restart nginx-app"')

@task
def notebook():
    local('docker exec -u="root" -it dva-server bash -c "pip install --upgrade jupyter"')
    local('docker exec -u="root" -it dva-server bash -c "jupyter notebook --allow-root"')