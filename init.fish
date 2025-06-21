#!/usr/bin/env fish

cd whope

set -x TOKENIZERS_PARALLELISM false 
set -x CUDA_VISIBLE_DEVICES -1
set -x DJANGO_SETTINGS_MODULE "whope.settings"
set -x TF_CPP_MIN_LOG_LEVEL 3
set -x TF_CPP_MIN_VLOG_LEVEL 3

echo "
use whope
db.dropDatabase()
.exit
" | mongosh --quiet

poetry run celery --app whope purge --force
pkill -9 -f celery
poetry run celery --app whope worker --loglevel warning &
poetry run daphne --verbosity 1 whope.asgi:application
