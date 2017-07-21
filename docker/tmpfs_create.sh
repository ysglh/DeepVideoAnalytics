#!/usr/bin/env bash
docker volume rm dvadata
docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=size=2g,uid=1000 dvadata