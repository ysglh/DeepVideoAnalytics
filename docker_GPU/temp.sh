#!/usr/bin/env bash
apt-get update && \
    apt-get install -y \
	nginx \
	supervisor \
pip install uwsgi
echo "daemon off;" >> /etc/nginx/nginx.conf
mv nginx-app.conf /etc/nginx/sites-available/default
mv supervisor-app.conf /etc/supervisor/conf.d/
python manage.py collectstatic --no-input
