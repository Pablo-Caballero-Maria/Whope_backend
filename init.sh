#!/bin/bash
cd whope_backend
export DJANGO_SETTINGS_MODULE=whope_backend.settings
poetry run daphne --verbosity 1 whope_backend.asgi:application
