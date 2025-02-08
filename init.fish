#!/bin/fish
cd whope

set -x DJANGO_SETTINGS_MODULE whope.settings
# set -x PYTHONOPTIMIZE 1

echo "
use whope
db.dropDatabase()
.exit
" | mongosh --quiet

poetry run celery --app whope purge --force
pkill -9 -f celery
poetry run celery --app whope worker --loglevel warning &
poetry run daphne --verbosity 1 whope.asgi:application
