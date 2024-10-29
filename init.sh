#!/bin/bash
fuser -k 8000/tcp
kill -9 $(jobs -p)
sleep 5
export TESTING=0
cd whope_backend
export DJANGO_SETTINGS_MODULE=whope_backend.settings
poetry run celery --app whope_backend purge --force
poetry run celery --app whope_backend worker --loglevel warning &
poetry run daphne --verbosity 1 whope_backend.asgi:application
