#!/bin/bash
cd whope

export DJANGO_SETTINGS_MODULE=whope.settings
export TF_CPP_MIN_LOG_LEVEL=1
export TF_CPP_MIN_VLOG_LEVEL=1

# mongosh --quiet <<EOF
#      use whope
#      db.dropDatabase()
#     .exit
# EOF

poetry run celery --app whope purge --force
pkill -9 -f celery
poetry run celery --app whope worker --loglevel warning &
poetry run daphne --verbosity 1 whope.asgi:application
