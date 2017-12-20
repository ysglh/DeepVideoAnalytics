Deployment with Kubernetes on Google Cloud Engine (Ongoing)
===

## To Do:

[x] Port database, rabbitmq and webserver
[x] Automatically create an admin users via environment variable
[ ] Add google-cloud-storage python lib and update fs.py
[ ] Add code to automatically create cluster
[ ] Add code to automatically create media and static asset bucket
[ ] Enable / add example for HTTPS ingress  
[ ] Add code to auto-scale cluster using pre-emptible VM node pool
[ ] Use helm charts or at least statefulset for Postgres & RabbitMQ.
[ ] Port DeepVideoAnalytics.com from AWS/Heroku to GKE.

### Deployment steps:

0. Ensure that GKE cluster is setup and authed by following instructions from GCP console.
 
1. Create config.py with values for secrets_template.yml

2. Run "create_bucket.py" to create and make Google Cloud storage bucket public (used for storing media assets.)

3. Run "create_secrets.py" to create secrets.yml

4. Run "launch.sh" to launch containers.