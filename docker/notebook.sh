#!/usr/bin/env bash
docker exec -u="root" -it dva-server bash -c "pip install --upgrade jupyter"
docker exec -u="root" -it dva-server bash -c "jupyter notebook --allow-root"
