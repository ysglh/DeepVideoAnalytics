Deployment with Kubernetes on Google Cloud Engine (Ongoing)
===

### Deployment steps:

0. Ensure that GKE cluster is setup and auth by following instructions from GCP console.
 
1. Create config.py (fill & rename config_example.py) which contains values for secrets_template.yml

2. Run "create_bucket.py" to create and make Google Cloud storage bucket public (used for storing media assets.)

3. Run "create_secrets.py" to create secrets.yml

4. Run "launch.sh" to launch containers.

5. Open kubeproxy and visit external IP in services.


### Remaining tasks:    

[ ] Ensure that Postgres and RabbitMQ are "Stateful sets" / consider reusing a Helm Chart. 
   
[ ] Create a tutorial to demonstrate how to deploy DVA on GCP using only Google cloud shell. 

[ ] Support GPU containters.
     
[ ] Enable / add example for HTTP/HTTPS ingress and create seperate multi-region bucket to serve static files. 
  
[ ] Port DeepVideoAnalytics.com from AWS/Heroku to GKE.
  
[ ] Add code to auto-scale cluster using pre-emptible VM node pool.  