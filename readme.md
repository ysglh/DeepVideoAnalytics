#Deep Video Analytics
#### (Under development, please come back for updates)

### Installation using docker-compose

````bash
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalytics 
cd DeepVideoAnalytics/docker
docker-compose up 
````

### Distributed architecture

- Metadata stored in Postgres.
- Operations (Querying, Frame extraction & Indexing) performed using celery tasks and RabbitMQ.
- Separate queues and workers to allow select machines with GPU & RAM for indexing / computing features.
- Videos, frames, indexes, numpy vectors stored in media directory.

# Copyright
Copyright, Akshay Bhat, Cornell University, All rights reserved.