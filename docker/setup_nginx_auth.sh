#!/usr/bin/env bash
rm .htpasswd
echo -n 'dvauser:' >> .htpasswd
openssl passwd -apr1 >> .htpasswd
docker cp .htpasswd dva-server:/etc/nginx/.htpasswd
docker exec -u="root" -it dva-server "service nginx restart"
