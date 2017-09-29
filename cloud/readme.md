# Distributed Multi machine deployment on AWS & Heroku

This directory contains scripts and files required for a distributed deployment.

![Architecture](cloud.png "Distributed cloud architecture")

Checklist for cloud deployment 
(Incomplete, will add a cloudformation script in future)

- Setup Heroku app & Heroku postgres database
- Add Cloud RabbitMQ extension to heroku app
- Create two S3 buckets one for media & one for static assets
- Set correct permissions for both buckets using fab task
- Create cloudfront distribution in front of static assets
- Generate heroku.env which contains SECRET (for Django), RabbitMQ, media bucket name and Postgres URL
- Store heroku.env a private separate config S3 bucket
- Create AWS IAM roles, Security Groups
- Create EFS using web console, !Ensure that VPC Security Group inbound rules are correctly set to allow EC2 <--> EFS!

E.g. following inbound rules
```
Type        | Protocol Port Range   | Source
ALL Traffic | ALL ALL               | sg-(Security group of VPC)
NFS (2049)  | TCP (6) 2049          | sg-(Security group of EC2)
```
- Add information to config_example.py and rename it to config.py
- Perform heroku bootstrap task and deploy to heroku
- Launch AWS fleet
