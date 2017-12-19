#!/usr/bin/env bash
source ~/aws.env && docker-compose -f docker-compose-linode.yml down -v
source ~/aws.env && docker-compose -f docker-compose-linode.yml up -d
