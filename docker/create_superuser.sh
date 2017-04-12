#!/usr/bin/env bash
docker exec -u="root" -it dva-server python manage.py createsuperuser
