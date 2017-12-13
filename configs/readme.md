### supervised, nginx & uwsgi config

Following files are used by `fab init_server` when a server container is launched during the startup phase.
They configure nginx, uwsgi and superviser running inside the container. nginx is used to serve media and static
files when running in single machine mode. When deployed on cloud the media and static assets are served through
AWS S3/Cloudfront (only for static assets) or GCS.

- {nginx-app.conf, nginx-app_password.conf, nginx.conf, supervisor-app.conf}  
- {uwsgi.ini, uwsgi_params}