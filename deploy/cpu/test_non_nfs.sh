#!/usr/bin/env bash
source ~/aws.env && docker-compose -f docker-compose-non-nfs.yml down -v
source ~/aws.env && docker-compose -f docker-compose-non-nfs.yml up -d