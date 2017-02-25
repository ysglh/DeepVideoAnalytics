# Architecture & Backups

## Deploying on multiple machines

As an example consider the case where you have

- Webserver on Heroku
- Running a sepearte RabbitMQ & Postgres is easy since docker container uses environment variables to connect.

### Sharing media folder

Two solutions:

#### 1. Using NFS or AWS EFS (slow without prefetching)

#### 2. Using AWS S3 and prefetching images

### Re-routing requests for /media/

Ideally django filter etc. when creating those urls within templates.

Temporary solution simply redirect to S3 bucket.
