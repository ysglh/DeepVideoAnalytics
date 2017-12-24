from fabric.api import task, local


@task
def shell(container_name="dva-server"):
    local('docker exec -u="root" -it {} bash'.format(container_name))

@task
def notebook():
    local('docker exec -u="root" -it dva-server bash -c "jupyter notebook --allow-root"')