#!/usr/bin/env bash
set -xe
mkdir ~/.aws

cat >> ~/.aws/credentials << EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF
cat ~/.aws/credentials
aws s3 cp s3://aub3config/.netrc ~/.netrc # using AWS credentials stored in environment variables copy .netrc from private bucket
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo # clone private repo
mv DeepVideoAnalyticsDemo dvap # mv cloned repo
rm ~/.netrc # remove .netrc
rm -rf ~/.aws