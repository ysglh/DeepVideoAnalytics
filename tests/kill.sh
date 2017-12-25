#!/usr/bin/env sh
ps auxww | grep 'celery -A dva * ' | awk '{print $2}' | xargs kill -9