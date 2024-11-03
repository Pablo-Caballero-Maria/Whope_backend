#!/bin/bash
cd whope
export DJANGO_SETTINGS_MODULE=whope.settings
poetry run celery --app whope worker --loglevel warning &
poetry run daphne --verbosity 1 whope.asgi:application
