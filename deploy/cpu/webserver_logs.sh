#!/usr/bin/env bash
docker exec -u="root" -it webserver bash -c 'cat /var/log/supervisor/app-*'