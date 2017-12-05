#!/usr/bin/env bash
aws s3 cp s3://aub3config/.netrc ~/.netrc
git remote add private https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsPrivate
git pull private master