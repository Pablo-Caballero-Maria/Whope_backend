#!/bin/bash
cd whope
export DJANGO_SETTINGS_MODULE=whope.settings

mongosh --quiet <<EOF
use whope
db.dropDatabase()
.exit
EOF

poetry run celery --app whope purge --force
poetry run celery --app whope worker --loglevel warning &
poetry run daphne --verbosity 1 whope.asgi:application
