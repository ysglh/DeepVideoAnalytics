#!/usr/bin/env shell
docker exec -u="root" -it webserver bash -c 'cat /var/log/supervisor/app-*'