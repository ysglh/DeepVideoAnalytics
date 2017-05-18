#!/usr/bin/env bash
set -xe
aws s3 cp s3://aub3config/.netrc .netrc # using AWS credentials stored in environment variables copy .netrc from private bucket
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo # clone private repo
mv DeepVideoAnalyticsDemo dvap # mv cloned repo
rm .netrc # remove .netrc