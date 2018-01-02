#!/usr/bin/env bash
docker-compose -f docker-compose-non-nfs.yml down -v
docker-compose -f docker-compose-non-nfs.yml up -d