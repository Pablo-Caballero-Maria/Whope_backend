#!/bin/bash
export TESTING=0
cd whope_backend
export DJANGO_SETTINGS_MODULE=whope_backend.settings
poetry run daphne whope_backend.asgi:application
