import os
import config
command = 'gcloud beta container --project "{project_name}" node-pools create "{pool_name}" --zone "{zone}" --cluster "{cluster_name}" ' \
          '--machine-type "n1-standard-2" --image-type "COS" ' \
          '--disk-size "100" ' \
          '--scopes "https://www.googleapis.com/auth/compute","https://www.googleapis.com/auth/devstorage.read_write",' \
          '"https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring",' \
          '"https://www.googleapis.com/auth/servicecontrol",' \
          '"https://www.googleapis.com/auth/service.management.readonly",' \
          '"https://www.googleapis.com/auth/trace.append" ' \
          '--preemptible --num-nodes "{count}"  '
if __name__ == '__main__':
    command = command.format(project_name=config.project_name,
                         pool_name="premptpool",cluster_name=config.cluster_name,
                         zone=config.zone,count=5)
    print command
    os.system(command)