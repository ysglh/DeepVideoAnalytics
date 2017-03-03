#To Do list Alpha version

- [x] Django App
- [x] Tasks using Celery & RabbitMQ
- [x] Postgres database
- [x] Deployment using docker-compose
- [x] Minimal user interface for uploading and browsing uploaded videos/images
- [x] Task for frame extraction from videos
- [x] Simple detection models using Darknet YOLO
- [x] Working visual search & indexer tasks using PyTorch
- [X] Simple set of tests (E.g. upload a video, perform processing, indexing, detection)
- [X] Deployment using nvidia-docker-compose for machines with GPU
- [X] Continuous integration test suite
- [X] Improved user interface for browsing past queries
- [X] Improve TEvent model to track state of tasks
- [X] Improved frame extraction using PySceneDetect (every 100th frame and frame selected by content change)
- [X] Integrate Tensorflow 1.0
- [X] Improved models by adding information about user performing the uploading video/dataset
- [X] Automated docker based testing
- [X] Implement a method to backup postgres db & media folder to S3 via a single command
- [X] Integrate youtube-dl for downloading videos
- [X] Test Deployment on AWS P2 machines running nvidia-docker 
- [X] Implemented nginx <-> uwsgi <-> django on GPU container for optimized serving of videos and static assets.
- [ ] Implement smarter index updates via a Broadcast task (update_index) that gets sent to all workers on indexer queue 
- [ ] Index detected objects

#To Do list 0.1 version
- [ ] Attach Open Face Docker container
- [ ] Enable manual restart of tasks via status page
- [ ] Cloudformation template for use with ECS & EFS
- [ ] Improve OpenCV video processing
