#!/usr/bin/env bash
docker exec -u="root" -it dva-server bash -c "pip install --upgrade awscli && aws configure"