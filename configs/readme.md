### supervised, nginx & uwsgi config

Following files are used by `fab init_server` when a server container is launched during the startup phase.
They configure nginx, uwsgi and superviser running inside the container. nginx is used to serve media and static
files when running in single machine mode. When deployed on Heroku the media and static assets are served through
AWS S3 and Cloudfront (only for static assets).

- {nginx-app.conf, nginx-app_password.conf, nginx.conf, supervisor-app.conf}  
- {uwsgi.ini, uwsgi_params}  

### Deep Video Analytics config

- models_extra.json & models.json   
  (by default models specified in models.json are loaded during fab init_models)
- /templates  
  (contains example DVAPQL templates, which are stored in the database)
- vdn_servers.json  
  (contains VDN servers to pull additional datasets and models.)