### Docker installation for use with nvidia-docker

Please install https://github.com/eywalker/nvidia-docker-compose and replace 'docker-compose with nvidia-docker-compose in all commands.

E.g. to launch the stack just run 
````bash
nvidia-docker-compose up
````
And open port 8000 on localhost once the server is up (typically wait couple of minutes for all containers to be ready and django migrations to be applied.)
