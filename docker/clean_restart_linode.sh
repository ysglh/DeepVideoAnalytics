#!/usr/bin/env bash
docker-compose -f docker-compose-linode.yml down -v
docker-compose -f docker-compose-linode.yml up