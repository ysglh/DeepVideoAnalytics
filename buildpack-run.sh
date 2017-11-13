#!/usr/bin/env bash
set -e
export "AWS_SECRET_ACCESS_KEY=$(cat $ENV_DIR/AWS_SECRET_ACCESS_KEY)"
export "AWS_ACCESS_KEY_ID=$(cat $ENV_DIR/AWS_ACCESS_KEY_ID)"
#aws s3 cp s3://aub3config/.netrc ~/.netrc # using AWS credentials stored in environment variables copy .netrc from private bucket
#git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo # clone private repo
#mv DeepVideoAnalyticsDemo dvap # mv cloned repo
#rm ~/.netrc # remove .netrc
rm -rf ~/.aws