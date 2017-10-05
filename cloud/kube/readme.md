Deployment with Kubernetes on Google Cloud Engine
===

## Advantages

- Kubernetes is easier to use than ECS
- Integrated load balancing allowing expensive heroku to be replaced
- Far easier to auto-scale

## Issues

- Lack of AWS lambda equivalent (Google cloud functions only support NodeJS)
- Lack of EFS (There is no network filesystem)
- Expensive GPUs
- Much better I/O that EC2
  
## Decision

Stick with AWS+Heroku for now, until Google Cloud functions allows Python.

