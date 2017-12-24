#!/usr/bin/env bash
# get Github token from AWS
aws s3 cp s3://aub3config/.netrc ~/.netrc
# Add remote and pull repo
git remote add private https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsPrivate
git pull private master
rm ~/.netrc