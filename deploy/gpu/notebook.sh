#!/usr/bin/env bash
docker exec -u="root" -it webserver bash -c "cd / && ./run_jupyter.sh --allow-root"