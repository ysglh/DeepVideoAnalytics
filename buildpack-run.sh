#!/usr/bin/env bash
set -xe
export_env_dir() {
  env_dir=$ENV_DIR
  whitelist_regex=${2:-''}
  blacklist_regex=${3:-'^(PATH|GIT_DIR|CPATH|CPPATH|LD_PRELOAD|LIBRARY_PATH)$'}
  if [ -d "$env_dir" ]; then
    for e in $(ls $env_dir); do
      echo "$e" | grep -E "$whitelist_regex" | grep -qvE "$blacklist_regex" &&
      export "$e=$(cat $env_dir/$e)"
      :
    done
  fi
}
export_env_dir
aws s3 cp s3://aub3config/.netrc ~/.netrc # using AWS credentials stored in environment variables copy .netrc from private bucket
git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo # clone private repo
mv DeepVideoAnalyticsDemo dvap # mv cloned repo
rm ~/.netrc # remove .netrc
rm -rf ~/.aws