#!/usr/bin/env bash
source ~/aws.env && docker-compose -f docker-compose.yml down
source ~/aws.env && docker-compose -f docker-compose.yml up -d